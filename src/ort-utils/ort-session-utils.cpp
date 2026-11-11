#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

#ifdef _WIN32
#include <dml_provider_factory.h>
#include <wchar.h>
#endif // _WIN32

#include <obs-module.h>

#include "ort-session-utils.h"
#include "consts.h"

#if defined(__linux__)
namespace {

struct LinuxExecutionProviderAvailability {
  bool migraphx = false;
  bool tensorrt = false;
};

void setLinuxEnvironmentPathIfUnset(const char *variableName,
                                    const std::filesystem::path &path)
{
  if (std::getenv(variableName) != nullptr) {
    return;
  }
  const auto normalizedPath = path.lexically_normal().string();
  if (!normalizedPath.empty()) {
    setenv(variableName, normalizedPath.c_str(), 0);
  }
}

void setupLinuxAmdCaches()
{
  static std::once_flag cacheSetupOnce;
  std::call_once(cacheSetupOnce, []() {
    if (!std::filesystem::is_directory("/opt/rocm")) {
      return;
    }

    char *miopenPathRaw = obs_module_config_path("amd/miopen");
    char *migraphxPathRaw = obs_module_config_path("amd/migraphx");

    if (miopenPathRaw != nullptr) {
      const std::filesystem::path miopenPath(miopenPathRaw);
      std::error_code ec;
      std::filesystem::create_directories(miopenPath, ec);
      setLinuxEnvironmentPathIfUnset("MIOPEN_USER_DB_PATH", miopenPath);
      setLinuxEnvironmentPathIfUnset("MIOPEN_CUSTOM_CACHE_DIR", miopenPath);
      bfree(miopenPathRaw);
    }

    // MIGraphX can return a transposed view (e.g. NHWC strides on an NCHW-shaped
    // tensor) as a program output. ORT treats outputs as packed, so the values
    // land in the wrong positions (PPHumanSeg comes back scrambled).
    // standardize_outputs makes MIGraphX copy such outputs to a standard layout.
    // MIGraphX reads this once per process, so it must be set before any compile.
    const char *gpuOptions = std::getenv("MIGRAPHX_GPU_OPTIONS");
    if (gpuOptions == nullptr) {
      setenv("MIGRAPHX_GPU_OPTIONS", "{standardize_outputs:true}", 0);
    } else if (std::string(gpuOptions).find("standardize_outputs") == std::string::npos) {
      blog(LOG_WARNING,
           "MIGRAPHX_GPU_OPTIONS is set without standardize_outputs; models with "
           "transposed outputs (e.g. PPHumanSeg) may produce wrong masks on MIGraphX");
    }

    if (migraphxPathRaw != nullptr) {
      const std::filesystem::path migraphxPath(migraphxPathRaw);
      // The EP's MXR cache key covers neither MIGraphX backend options nor the
      // real MIGraphX version, so MXRs compiled elsewhere (or without
      // standardize_outputs) would be loaded as-is. Keep a private cache.
      const std::filesystem::path modelCachePath = migraphxPath / "standardized-outputs";
      std::error_code ec;
      // The EP aborts the session if the cache directory doesn't exist.
      std::filesystem::create_directories(modelCachePath, ec);
      setLinuxEnvironmentPathIfUnset("ORT_MIGRAPHX_CACHE_PATH", migraphxPath);
      setenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", modelCachePath.lexically_normal().c_str(), 1);
      blog(LOG_INFO, "MIGraphX compiled-model cache: %s", modelCachePath.c_str());
      bfree(migraphxPathRaw);
    }
  });
}

Ort::Env &getSharedOrtEnv()
{
  static Ort::Env env = []() {
    setupLinuxAmdCaches();
    return Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "obs-backgroundremoval");
  }();
  return env;
}

const LinuxExecutionProviderAvailability &getLinuxExecutionProviderAvailability()
{
  static const LinuxExecutionProviderAvailability availability = []() {
    // Ensure Ort::Env (and DefaultLogger) exists before loading provider libs.
    getSharedOrtEnv();

    LinuxExecutionProviderAvailability providers;
    try {
      const auto names = Ort::GetAvailableProviders();
      for (const auto &n : names) {
        if (n.find("MIGraphX") != std::string::npos) providers.migraphx = true;
        if (n.find("Tensorrt") != std::string::npos || n.find("TensorRT") != std::string::npos)
          providers.tensorrt = true;
      }
    } catch (const std::exception &e) {
      blog(LOG_WARNING, "Failed to enumerate ONNX Runtime providers: %s", e.what());
    }
    return providers;
  }();
  return availability;
}

// Loading MIGraphX's provider library pulls in libmigraphx_tf, which registers
// its own copy of TensorFlow's tensor_shape.proto into libprotobuf's global
// descriptor pool. If anything else in the OBS process has already registered
// a TF descriptor (libopencv_dnn, libtensorflow, libopenvino's TF frontend),
// the protobuf CHECK aborts the entire process. Detect that situation and skip
// MIGraphX so the user gets a fallback instead of a crash.
bool isProtobufConflictRisk(std::string &conflictingLib)
{
  static const char *const conflictMarkers[] = {
    "libopencv_dnn",
    "libtensorflow",
    "libtensorflow_framework",
    "libopenvino_tensorflow_frontend",
  };
  std::ifstream maps("/proc/self/maps");
  if (!maps.is_open()) return false;
  std::string line;
  while (std::getline(maps, line)) {
    for (const char *marker : conflictMarkers) {
      if (line.find(marker) != std::string::npos) {
        const auto slash = line.rfind('/');
        conflictingLib = (slash != std::string::npos) ? line.substr(slash + 1) : marker;
        return true;
      }
    }
  }
  return false;
}

bool isLinuxGpuProviderSelection(const std::string &useGPU)
{
  return useGPU == USEGPU_MIGRAPHX || useGPU == USEGPU_TENSORRT;
}

void resetOrtSessionState(filter_data *tf)
{
  tf->session.reset();
  tf->inputNames.clear();
  tf->outputNames.clear();
  tf->inputTensor.clear();
  tf->outputTensor.clear();
  tf->inputDims.clear();
  tf->outputDims.clear();
  tf->inputTensorValues.clear();
  tf->outputTensorValues.clear();
}

bool populateSessionTensors(filter_data *tf)
{
  if (tf->session.get() == nullptr || tf->model.get() == nullptr) {
    return false;
  }

  tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

  if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims)) {
    blog(LOG_ERROR, "Unable to get model input and output shapes");
    return false;
  }

  for (size_t i = 0; i < tf->inputNames.size(); i++) {
    blog(LOG_INFO, "Model %s input %d: name %s shape (%d dim) %d x %d x %d x %d",
         tf->modelSelection.c_str(), (int)i, tf->inputNames[i].get(), (int)tf->inputDims[i].size(),
         (int)tf->inputDims[i][0],
         ((int)tf->inputDims[i].size() > 1) ? (int)tf->inputDims[i][1] : 0,
         ((int)tf->inputDims[i].size() > 2) ? (int)tf->inputDims[i][2] : 0,
         ((int)tf->inputDims[i].size() > 3) ? (int)tf->inputDims[i][3] : 0);
  }
  for (size_t i = 0; i < tf->outputNames.size(); i++) {
    blog(LOG_INFO, "Model %s output %d: name %s shape (%d dim) %d x %d x %d x %d",
         tf->modelSelection.c_str(), (int)i, tf->outputNames[i].get(),
         (int)tf->outputDims[i].size(), (int)tf->outputDims[i][0],
         ((int)tf->outputDims[i].size() > 1) ? (int)tf->outputDims[i][1] : 0,
         ((int)tf->outputDims[i].size() > 2) ? (int)tf->outputDims[i][2] : 0,
         ((int)tf->outputDims[i].size() > 3) ? (int)tf->outputDims[i][3] : 0);
  }

  tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues,
                                   tf->inputTensorValues, tf->inputTensor, tf->outputTensor);
  return true;
}

void appendLinuxExecutionProvider(Ort::SessionOptions &sessionOptions,
                                  const std::string &providerName)
{
  if (providerName == USEGPU_MIGRAPHX) {
    std::string conflict;
    if (isProtobufConflictRisk(conflict)) {
      throw Ort::Exception(
        ("MIGraphX cannot be initialized because " + conflict +
         " is loaded in this process; loading MIGraphX's TF parser would "
         "duplicate-register protobuf descriptors and abort. Disable the "
         "conflicting plugin or pick a different provider.")
          .c_str(),
        ORT_FAIL);
    }
    // Zero-init leaves migraphx_mem_limit at 0; ORT's default is SIZE_MAX.
    OrtMIGraphXProviderOptions migraphxOptions{};
    migraphxOptions.device_id = 0;
    migraphxOptions.migraphx_fp16_enable = 0;
    migraphxOptions.migraphx_fp8_enable = 0;
    migraphxOptions.migraphx_int8_enable = 0;
    migraphxOptions.migraphx_use_native_calibration_table = 0;
    migraphxOptions.migraphx_int8_calibration_table_name = nullptr;
    migraphxOptions.migraphx_save_compiled_model = 0;
    migraphxOptions.migraphx_save_model_path = nullptr;
    migraphxOptions.migraphx_load_compiled_model = 0;
    migraphxOptions.migraphx_load_model_path = nullptr;
    migraphxOptions.migraphx_exhaustive_tune = false;
    migraphxOptions.migraphx_mem_limit = std::numeric_limits<size_t>::max();
    migraphxOptions.migraphx_arena_extend_strategy = 0;
    sessionOptions.AppendExecutionProvider_MIGraphX(migraphxOptions);
    blog(LOG_INFO, "Using ONNX Runtime execution provider: MIGraphX");
    return;
  }

  if (providerName == USEGPU_TENSORRT) {
    OrtTensorRTProviderOptions tensorrtOptions{};
    tensorrtOptions.device_id = 0;
    tensorrtOptions.trt_max_partition_iterations = 1000;
    tensorrtOptions.trt_min_subgraph_size = 1;
    tensorrtOptions.trt_max_workspace_size = 1U << 30;
    sessionOptions.AppendExecutionProvider_TensorRT(tensorrtOptions);
    blog(LOG_INFO, "Using ONNX Runtime execution provider: TensorRT");
    return;
  }
}

void configureSessionOptions(filter_data *tf, Ort::SessionOptions &sessionOptions)
{
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (tf->useGPU == USEGPU_CPU) {
    sessionOptions.SetInterOpNumThreads(tf->numThreads);
    sessionOptions.SetIntraOpNumThreads(tf->numThreads);
    return;
  }

  // GPU EPs (TensorRT, MIGraphX): sequential + no mem pattern, matching other platforms.
  sessionOptions.DisableMemPattern();
  sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
}

bool tryCreateCpuFallbackSession(filter_data *tf)
{
  try {
    if (tf->modelFilepathStorage.empty()) {
      blog(LOG_ERROR, "CPU fallback session creation failed: model path is empty");
      return false;
    }

    Ort::SessionOptions cpuSessionOptions;
    tf->useGPU = USEGPU_CPU;
    configureSessionOptions(tf, cpuSessionOptions);
    resetOrtSessionState(tf);
    tf->session.reset(
      new Ort::Session(getSharedOrtEnv(), tf->modelFilepathStorage.c_str(), cpuSessionOptions));
    if (!populateSessionTensors(tf)) {
      return false;
    }
    blog(LOG_WARNING, "Fell back to CPU execution provider");
    return true;
  } catch (const std::exception &e) {
    blog(LOG_ERROR, "CPU fallback session creation failed: %s", e.what());
    return false;
  }
}

} // namespace

const char *getLinuxGpuExecutionProviderValue()
{
  const auto &availability = getLinuxExecutionProviderAvailability();
  if (availability.migraphx) return USEGPU_MIGRAPHX;
  if (availability.tensorrt) return USEGPU_TENSORRT;
  return nullptr;
}

const char *getLinuxGpuExecutionProviderTextKey()
{
  const char *providerValue = getLinuxGpuExecutionProviderValue();
  if (providerValue == nullptr) return nullptr;
  return providerValue == std::string(USEGPU_MIGRAPHX) ? "GPUMIGraphX" : "GPUTensorRT";
}

std::string normalizeGpuProviderSelection(const std::string &useGPU)
{
  if (!isLinuxGpuProviderSelection(useGPU)) return useGPU;
  const char *providerValue = getLinuxGpuExecutionProviderValue();
  if (providerValue == nullptr) return USEGPU_CPU;
  return providerValue;
}
#endif

void createOrtSession(filter_data *tf)
{
  if (tf->model.get() == nullptr) {
    blog(LOG_ERROR, "Model object is not initialized");
    return;
  }

  char *modelFilepath_rawPtr = obs_module_file(tf->modelSelection.c_str());

  if (modelFilepath_rawPtr == nullptr) {
    blog(LOG_ERROR, "Unable to get model filename %s from plugin.", tf->modelSelection.c_str());
    return;
  }

  tf->modelFilepathStorage = modelFilepath_rawPtr;
  bfree(modelFilepath_rawPtr);

#if _WIN32
  tf->modelFilepathWStorage.assign(tf->modelFilepathStorage.begin(), tf->modelFilepathStorage.end());
  tf->modelFilepath = tf->modelFilepathWStorage.c_str();
#else
  tf->modelFilepath = tf->modelFilepathStorage.c_str();
#endif

  try {
#if defined(__linux__)
    tf->useGPU = normalizeGpuProviderSelection(tf->useGPU);
    Ort::SessionOptions sessionOptions;
    configureSessionOptions(tf, sessionOptions);
    resetOrtSessionState(tf);
    try {
      if (tf->useGPU != USEGPU_CPU) {
        appendLinuxExecutionProvider(sessionOptions, tf->useGPU);
      }
      tf->session.reset(new Ort::Session(getSharedOrtEnv(), tf->modelFilepath, sessionOptions));
    } catch (const std::exception &e) {
      blog(LOG_WARNING,
           "Failed to create Linux execution provider '%s' session: %s. Falling back to CPU.",
           tf->useGPU.c_str(), e.what());
      if (!tryCreateCpuFallbackSession(tf)) {
        return;
      }
      // tryCreateCpuFallbackSession already populated tensors
      return;
    }
#else
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (tf->useGPU != USEGPU_CPU) {
      sessionOptions.DisableMemPattern();
      sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    } else {
      sessionOptions.SetInterOpNumThreads(tf->numThreads);
      sessionOptions.SetIntraOpNumThreads(tf->numThreads);
    }
#endif
#ifdef _WIN32
    if (tf->useGPU == USEGPU_DML) {
      auto &api = Ort::GetApi();
      OrtDmlApi *dmlApi = nullptr;
      Ort::ThrowOnError(
        api.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dmlApi));
      Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
    }
#endif
#if defined(__APPLE__)
    if (tf->useGPU == USEGPU_COREML) {
      uint32_t coreml_flags = 0;
      coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
      Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags));
    }
#endif
#if !defined(__linux__)
    static Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "obs-backgroundremoval");
    resetOrtSessionState(tf);
    tf->session.reset(new Ort::Session(env, tf->modelFilepath, sessionOptions));
#endif
  } catch (const std::exception &e) {
    blog(LOG_ERROR, "%s", e.what());
    return;
  }

#if defined(__linux__)
  if (!populateSessionTensors(tf)) {
    return;
  }
#else
  Ort::AllocatorWithDefaultOptions allocator;

  tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

  if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims)) {
    blog(LOG_ERROR, "Unable to get model input and output shapes");
    return;
  }

  for (size_t i = 0; i < tf->inputNames.size(); i++) {
    blog(LOG_INFO, "Model %s input %d: name %s shape (%d dim) %d x %d x %d x %d",
         tf->modelSelection.c_str(), (int)i, tf->inputNames[i].get(), (int)tf->inputDims[i].size(),
         (int)tf->inputDims[i][0],
         ((int)tf->inputDims[i].size() > 1) ? (int)tf->inputDims[i][1] : 0,
         ((int)tf->inputDims[i].size() > 2) ? (int)tf->inputDims[i][2] : 0,
         ((int)tf->inputDims[i].size() > 3) ? (int)tf->inputDims[i][3] : 0);
  }
  for (size_t i = 0; i < tf->outputNames.size(); i++) {
    blog(LOG_INFO, "Model %s output %d: name %s shape (%d dim) %d x %d x %d x %d",
         tf->modelSelection.c_str(), (int)i, tf->outputNames[i].get(),
         (int)tf->outputDims[i].size(), (int)tf->outputDims[i][0],
         ((int)tf->outputDims[i].size() > 1) ? (int)tf->outputDims[i][1] : 0,
         ((int)tf->outputDims[i].size() > 2) ? (int)tf->outputDims[i][2] : 0,
         ((int)tf->outputDims[i].size() > 3) ? (int)tf->outputDims[i][3] : 0);
  }

  // Allocate buffers
  tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues,
                                   tf->inputTensorValues, tf->inputTensor, tf->outputTensor);
#endif
}

bool runFilterModelInference(filter_data *tf, const cv::Mat &imageBGRA, cv::Mat &output)
{
  if (tf->session.get() == nullptr) {
    // Onnx runtime session is not initialized. Problem in initialization
    return false;
  }
  if (tf->model.get() == nullptr) {
    // Model object is not initialized
    return false;
  }

  // To RGB
  cv::Mat imageRGB;
  cv::cvtColor(imageBGRA, imageRGB, cv::COLOR_BGRA2RGB);

  // Resize to network input size
  uint32_t inputWidth, inputHeight;
  tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

  cv::Mat resizedImageRGB;
  cv::resize(imageRGB, resizedImageRGB, cv::Size(inputWidth, inputHeight));

  // Prepare input to nework
  cv::Mat resizedImage, preprocessedImage;
  resizedImageRGB.convertTo(resizedImage, CV_32F);

  tf->model->prepareInputToNetwork(resizedImage, preprocessedImage);

  tf->model->loadInputToTensor(preprocessedImage, inputWidth, inputHeight, tf->inputTensorValues);

  // Run network inference (with one-shot GPU→CPU fallback on Linux)
  try {
    tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames, tf->inputTensor,
                                   tf->outputTensor);
  } catch (const Ort::Exception &e) {
#if defined(__linux__)
    if (tf->useGPU != USEGPU_CPU) {
      blog(LOG_WARNING,
           "GPU inference failed with provider '%s': %s. Falling back to CPU.",
           tf->useGPU.c_str(), e.what());
      if (!tryCreateCpuFallbackSession(tf)) {
        return false;
      }
      // New session has new buffers; reload inputs and retry once.
      tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);
      tf->model->loadInputToTensor(preprocessedImage, inputWidth, inputHeight,
                                   tf->inputTensorValues);
      try {
        tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames,
                                       tf->inputTensor, tf->outputTensor);
      } catch (const Ort::Exception &retryEx) {
        blog(LOG_ERROR, "CPU fallback inference also failed: %s", retryEx.what());
        return false;
      }
    } else {
      blog(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
      return false;
    }
#else
    blog(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
    return false;
#endif
  }

  // Get output
  // Map network output to cv::Mat
  cv::Mat outputImage = tf->model->getNetworkOutput(tf->outputDims, tf->outputTensorValues);

  // Assign output to input in some models that have temporal information
  tf->model->assignOutputToInput(tf->outputTensorValues, tf->inputTensorValues);

  // Post-process output. The image will now be in [0,1] float, BHWC format
  tf->model->postprocessOutput(outputImage);

  // Convert [0,1] float to CV_8U [0,255]
  outputImage.convertTo(output, CV_8U, 255.0);

  return true;
}
