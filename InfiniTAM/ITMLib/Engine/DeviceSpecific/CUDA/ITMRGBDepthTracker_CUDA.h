#pragma once

#include "../../ITMRGBDepthTracker.h"

namespace ITMLib {
namespace Engine {

class ITMRGBDepthTracker_CUDA : public ITMRGBDepthTracker {
 private:
  Vector2f* f_device; float* g_device; float* h_device;
  Vector2f* f_host; float* g_host; float* h_host;

 public:
  virtual void F_oneLevel(float* f, ITMPose* toRGBPose);
  virtual void G_oneLevel(float* gradient, float* hessian, ITMPose* toRGBPose, ITMPose* toDepthPose) const;

  ITMRGBDepthTracker_CUDA(Vector2i imgSize, TrackerIterationType* trackingRegime, int noHierarchyLevels,
                          const ITMLowLevelEngine* lowLevelEngine);
  ~ITMRGBDepthTracker_CUDA();
};

}
}
