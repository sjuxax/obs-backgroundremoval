#ifndef MODELSINET_H
#define MODELSINET_H

#include "Model.h"

class ModelSINET : public ModelBCHW {
  public:
  ModelSINET(/* args */) {}
  ~ModelSINET() {}

  virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
  {
    cv::subtract(resizedImage, cv::Scalar(102.890434, 111.25247, 126.91212), resizedImage);
    cv::multiply(resizedImage, cv::Scalar(1.0 / 62.93292, 1.0 / 62.82138, 1.0 / 66.355705) / 255.0,
                 resizedImage);
    hwc_to_chw(resizedImage, preprocessedImage);
  }

  virtual void postprocessOutput(cv::Mat &output)
  {
    // Softmax is 2-class BCHW (background, person). Convert to HWC then keep person.
    ModelBCHW::postprocessOutput(output);
    std::vector<cv::Mat> channels;
    cv::split(output, channels);
    if (channels.size() >= 2) {
      output = channels[1];
    } else if (!channels.empty()) {
      output = channels[0];
    }
  }
};

#endif // MODELSINET_H
