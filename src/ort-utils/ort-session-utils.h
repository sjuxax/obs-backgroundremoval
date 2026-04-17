#ifndef ORT_SESSION_UTILS_H
#define ORT_SESSION_UTILS_H

#include <opencv2/core/types.hpp>
#include <string>

#include "FilterData.h"

void createOrtSession(filter_data *tf);

bool runFilterModelInference(filter_data *tf, const cv::Mat &imageBGRA, cv::Mat &output);

#if defined(__linux__)
const char *getLinuxGpuExecutionProviderValue();
const char *getLinuxGpuExecutionProviderTextKey();
std::string normalizeGpuProviderSelection(const std::string &useGPU);
#endif

#endif /* ORT_SESSION_UTILS_H */
