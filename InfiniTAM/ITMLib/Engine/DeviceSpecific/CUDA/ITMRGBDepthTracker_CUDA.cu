#include "ITMRGBDepthTracker_CUDA.h"

#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMRGBDepthTracker.h"
#include "../../DeviceAgnostic/ITMPixelUtils.h"
#include "../../../../ORUtils/CUDADefines.h"

using namespace ITMLib::Engine;

__global__
void RGBDepthTrackerOneLevel_f_device(Vector2f* f_device,
                                      const Vector4f* locations, const Vector4f* colours, const Vector4u* rgb,
                                      Vector2i rgbImageSize, Vector4f projParams, Matrix4f M, int noTotalPoints);

__global__
void RGBDepthTrackerOneLevel_g_rt_device(float* g_device, float* h_device,
                                         const Vector4f* locations, const Vector4f* colours,
                                         const Vector4u* rgb, const float* depth,
                                         const Vector4s* gx, const Vector4s* gy,
                                         const float* gdx, const float* gdy,
                                         Vector2i rgbImageSize, Vector2i depthImageSize,
                                         Vector4f projRGBParams, Vector4f projDepthParams,
                                         Matrix4f toRGBM, Matrix4f toDepthM, int noTotalPoints);

__global__
void RGBDepthTrackerOneLevel_g_ro_device(float* g_device, float* h_device,
                                         const Vector4f* locations, const Vector4f* colours,
                                         const Vector4u* rgb, const float* depth,
                                         const Vector4s* gx, const Vector4s* gy,
                                         const float* gdx, const float* gdy,
                                         Vector2i rgbImageSize, Vector2i depthImageSize,
                                         Vector4f projRGBParams, Vector4f projDepthParams,
                                         Matrix4f toRGBM, Matrix4f toDepthM, int noTotalPoints);

ITMRGBDepthTracker_CUDA::ITMRGBDepthTracker_CUDA(Vector2i imgSize,
                                                 TrackerIterationType* trackingRegime,
                                                 int noHierarchyLevels,
                                                 const ITMLowLevelEngine* lowLevelEngine)
    : ITMRGBDepthTracker(imgSize, trackingRegime, noHierarchyLevels, lowLevelEngine, MEMORYDEVICE_CUDA) {
  const int numPara = 6;
  const int numParaSQ = 6 + 5 + 4 + 3 + 2 + 1;

  ITMSafeCall(cudaMalloc((void**)&f_device, sizeof(Vector2f) * imgSize.x * imgSize.y / 128));
  ITMSafeCall(cudaMalloc((void**)&g_device, sizeof(float) * numPara * (imgSize.x * imgSize.y / 128)));
  ITMSafeCall(cudaMalloc((void**)&h_device, sizeof(float) * numParaSQ * (imgSize.x * imgSize.y / 128)));

  f_host = new Vector2f[imgSize.x * imgSize.y / 128];
  g_host = new float[numPara * (imgSize.x * imgSize.y / 128)];
  h_host = new float[numParaSQ * (imgSize.x * imgSize.y / 128)];
}

ITMRGBDepthTracker_CUDA::~ITMRGBDepthTracker_CUDA() {
  ITMSafeCall(cudaFree(f_device));
  ITMSafeCall(cudaFree(g_device));
  ITMSafeCall(cudaFree(h_device));

  delete [] f_host;
  delete [] g_host;
  delete [] h_host;
}

void ITMRGBDepthTracker_CUDA::F_oneLevel(float* f, ITMPose* toRGBPose) {
  int noTotalPoints = trackingState->pointCloud->noTotalPoints;

  Vector4f projParams = view->calib->intrinsics_rgb.projectionParamsSimple.all;
  projParams.x /= 1 << levelId; projParams.y /= 1 << levelId;
  projParams.z /= 1 << levelId; projParams.w /= 1 << levelId;
  Vector2i rgbImageSize = viewHierarchy->levels[levelId]->rgb->noDims;
  Matrix4f M = toRGBPose->GetM();

  Vector4f *locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
  Vector4f *colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
  Vector4u *rgb = viewHierarchy->levels[levelId]->rgb->GetData(MEMORYDEVICE_CUDA);

  dim3 blockSize(128);
  dim3 gridSize((int)ceil(noTotalPoints * 1.0 / 128));

  memset(f_host, 0, sizeof(Vector2f) * gridSize.x);
  ITMSafeCall(cudaMemset(f_device, 0, sizeof(Vector2f) * gridSize.x));
  RGBDepthTrackerOneLevel_f_device<<<gridSize, blockSize>>>(f_device, locations, colours, rgb,
                                                            rgbImageSize, projParams, M, noTotalPoints);
  ITMSafeCall(cudaMemcpy(f_host, f_device, sizeof(Vector2f) * gridSize.x, cudaMemcpyDeviceToHost));

  float final_f = 0;
  countedPoints_valid = 0;
  for (size_t i = 0; i < gridSize.x; ++i) {
    if (f_host[i].y > 0) {
      final_f += f_host[i].x;
      countedPoints_valid += (int)f_host[i].y;
    }
  }

  float scaleForOcclusions = 0;
  if (countedPoints_valid == 0) {
    final_f = MY_INF;
    scaleForOcclusions = 1.0;
  } else {
    scaleForOcclusions = (float)noTotalPoints / countedPoints_valid;
  }

  f[0] = final_f * scaleForOcclusions;
}

void ITMRGBDepthTracker_CUDA::G_oneLevel(float* gradient, float* hessian,
                                         ITMPose* toRGBPose, ITMPose* toDepthPose) const {
  int noTotalPoints = trackingState->pointCloud->noTotalPoints;

  Vector4f projRGBParams = view->calib->intrinsics_rgb.projectionParamsSimple.all;
  Vector4f projDepthParams = view->calib->intrinsics_d.projectionParamsSimple.all;
  projRGBParams.x /= 1 << levelId; projRGBParams.y /= 1 << levelId;
  projRGBParams.z /= 1 << levelId; projRGBParams.w /= 1 << levelId;
  projDepthParams.x /= 1 << levelId; projDepthParams.y /= 1 << levelId;
  projDepthParams.z /= 1 << levelId; projDepthParams.w /= 1 << levelId;

  Matrix4f toRGBM = toRGBPose->GetM();
  Matrix4f toDepthM = toDepthPose->GetM();

  Vector2i rgbImageSize = viewHierarchy->levels[levelId]->rgb->noDims;
  Vector2i depthImageSize = viewHierarchy->levels[levelId]->depth->noDims;

  bool rotationOnly = (iterationType == TRACKER_ITERATION_ROTATION);
  int numPara = rotationOnly ? 3 : 6;
  int numParaSQ = rotationOnly ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;

  Vector4f *locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
  Vector4f *colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
  Vector4u *rgb = viewHierarchy->levels[levelId]->rgb->GetData(MEMORYDEVICE_CUDA);
  Vector4s *gx = viewHierarchy->levels[levelId]->gradientX_rgb->GetData(MEMORYDEVICE_CUDA);
  Vector4s *gy = viewHierarchy->levels[levelId]->gradientY_rgb->GetData(MEMORYDEVICE_CUDA);
  float *depth = viewHierarchy->levels[levelId]->depth->GetData(MEMORYDEVICE_CUDA);
  float *gdx = viewHierarchy->levels[levelId]->gradientX_depth->GetData(MEMORYDEVICE_CUDA);
  float *gdy = viewHierarchy->levels[levelId]->gradientY_depth->GetData(MEMORYDEVICE_CUDA);

  dim3 blockSize(128);
  dim3 gridSize((int)ceil(noTotalPoints * 1.0 / 128));

  memset(g_host, 0, sizeof(float) * numPara * gridSize.x);
  memset(h_host, 0, sizeof(float) * numParaSQ * gridSize.x);
  ITMSafeCall(cudaMemset(g_device, 0, sizeof(float) * numPara * gridSize.x));
  ITMSafeCall(cudaMemset(h_device, 0, sizeof(float) * numParaSQ * gridSize.x));
  if (rotationOnly) {
    RGBDepthTrackerOneLevel_g_ro_device<<<gridSize, blockSize>>>(g_device, h_device,
                                                                 locations, colours,
                                                                 rgb, depth,
                                                                 gx, gy, gdx, gdy,
                                                                 rgbImageSize, depthImageSize,
                                                                 projRGBParams, projDepthParams,
                                                                 toRGBM, toDepthM, noTotalPoints);
  } else {
    RGBDepthTrackerOneLevel_g_rt_device<<<gridSize, blockSize>>>(g_device, h_device,
                                                                 locations, colours,
                                                                 rgb, depth,
                                                                 gx, gy, gdx, gdy,
                                                                 rgbImageSize, depthImageSize,
                                                                 projRGBParams, projDepthParams,
                                                                 toRGBM, toDepthM, noTotalPoints);
  }
  ITMSafeCall(cudaMemcpy(g_host, g_device, sizeof(float) * numPara * gridSize.x, cudaMemcpyDeviceToHost));
  ITMSafeCall(cudaMemcpy(h_host, h_device, sizeof(float) * numParaSQ * gridSize.x, cudaMemcpyDeviceToHost));

  for (size_t blockId = 1; blockId < gridSize.x; ++blockId) {
    for (int i = 0; i < numPara; ++i)
      g_host[i] += g_host[blockId * numPara + i];
    for (int i = 0; i < numParaSQ; ++i)
      h_host[i] += h_host[blockId * numParaSQ + i];
  }

  float scaleForOcclusions = (float)noTotalPoints / countedPoints_valid;
  if (countedPoints_valid == 0)
    scaleForOcclusions = 1;

  // Expand the matrice
  for (int para = 0, counter = 0; para < numPara; para++) {
    gradient[para] = g_host[para] * scaleForOcclusions;
    for (int col = 0; col <= para; col++, counter++)
      hessian[para + col * numPara] = h_host[counter] * scaleForOcclusions;
  }
  for (int row = 0; row < numPara; row++) {
    for (int col = row + 1; col < numPara; col++)
      hessian[row + col * numPara] = hessian[col + row * numPara];
  }
}

__global__
void RGBDepthTrackerOneLevel_f_device(
    Vector2f* f_device,
    const Vector4f* locations, const Vector4f* colours, const Vector4u* rgb,
    Vector2i rgbImageSize, Vector4f projParams, Matrix4f M, int noTotalPoints) {
  int locId_local = threadIdx.x;
  int locId_global = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ ORUtils::Vector2_<float> f_shared[128];

  if (locId_global < noTotalPoints) {
    float colorDiffSq = getColorDifferenceSq(locations, colours, rgb, rgbImageSize, locId_global, projParams, M);
    if (colorDiffSq >= 0) {
      f_shared[locId_local].x = colorDiffSq;
      f_shared[locId_local].y = 1;
    } else {
      f_shared[locId_local].x = 0;
      f_shared[locId_local].y = 0;
    }
  } else {
    f_shared[locId_local].x = 0;
    f_shared[locId_local].y = 0;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (locId_local < offset) {
      f_shared[locId_local].x += f_shared[offset + locId_local].x;
      f_shared[locId_local].y += f_shared[offset + locId_local].y;
    }
    __syncthreads();
  }

  if (locId_local == 0) {
    f_device[blockIdx.x].x = f_shared[locId_local].x;
    f_device[blockIdx.x].y = f_shared[locId_local].y;
  }
}

__global__
void RGBDepthTrackerOneLevel_g_rt_device(float* g_device, float* h_device,
                                         const Vector4f* locations, const Vector4f* colours,
                                         const Vector4u* rgb, const float* depth,
                                         const Vector4s* gx, const Vector4s* gy,
                                         const float* gdx, const float* gdy,
                                         Vector2i rgbImageSize, Vector2i depthImageSize,
                                         Vector4f projRGBParams, Vector4f projDepthParams,
                                         Matrix4f toRGBM, Matrix4f toDepthM, int noTotalPoints) {
  int locId_local = threadIdx.x;
  int locId_global = threadIdx.x + blockIdx.x * blockDim.x;

  const int startPara = 0;
  const int numPara = 6;
  const int numParaSQ = 6 + 5 + 4 + 3 + 2 + 1;
  __shared__ float g_h_shared[numParaSQ * 128];

  // Compute Hessian matrix first
  float localGradient[numPara];
  float localHessian[numParaSQ];
  memset(localGradient, 0, sizeof(float) * numPara);
  memset(localHessian, 0, sizeof(float) * numParaSQ);
  if (locId_global < noTotalPoints) {
    computePerPointGH_rt_RGBDepth(localGradient, localHessian,
                                  locations, colours, rgb, depth,
                                  rgbImageSize, depthImageSize, locId_global,
                                  projRGBParams, projDepthParams, toRGBM, toDepthM,
                                  gx, gy, gdx, gdy, numPara, startPara);
    for (int i = 0; i < numParaSQ; ++i) 
      g_h_shared[locId_local * numParaSQ + i] = localHessian[i];
  } else {
    memset(g_h_shared + locId_local * numPara, 0, sizeof(float) * numParaSQ);
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (locId_local < offset) {
      for (int i = 0; i < numParaSQ; ++i)
        g_h_shared[locId_local * numParaSQ + i] += g_h_shared[(offset + locId_local) * numParaSQ + i];
    }
    __syncthreads();
  }

  if (locId_local == 0) {
    for (int i = 0; i < numParaSQ; ++i)
      h_device[blockIdx.x * numParaSQ + i] = g_h_shared[i];
  }
  __syncthreads();

  // Compute gradient
  if (locId_global < noTotalPoints) {
    for (int i = 0; i < numPara; ++i)
      g_h_shared[locId_local * numPara + i] = localGradient[i];
  } else {
    memset(g_h_shared + locId_local * numPara, 0, sizeof(float) * numPara);
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (locId_local < offset) {
      for (int i = 0; i < numPara; ++i)
        g_h_shared[locId_local * numPara + i] += g_h_shared[(offset + locId_local) * numPara + i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < numPara; ++i)
      g_device[blockIdx.x * numPara + i] = g_h_shared[i];
  }
}

__global__
void RGBDepthTrackerOneLevel_g_ro_device(float* g_device, float* h_device,
                                         const Vector4f* locations, const Vector4f* colours,
                                         const Vector4u* rgb, const float* depth,
                                         const Vector4s* gx, const Vector4s* gy,
                                         const float* gdx, const float* gdy,
                                         Vector2i rgbImageSize, Vector2i depthImageSize,
                                         Vector4f projRGBParams, Vector4f projDepthParams,
                                         Matrix4f toRGBM, Matrix4f toDepthM, int noTotalPoints) {
  int locId_local = threadIdx.x;
  int locId_global = threadIdx.x + blockIdx.x * blockDim.x;

  const int startPara = 3;
  const int numPara = 3;
  const int numParaSQ = 3 + 2 + 1;
  __shared__ float g_h_shared[numParaSQ * 128];

  // Compute Hessian matrix first
  float localGradient[numPara];
  float localHessian[numParaSQ];
  memset(localGradient, 0, sizeof(float) * numPara);
  memset(localHessian, 0, sizeof(float) * numParaSQ);
  if (locId_global < noTotalPoints) {
    computePerPointGH_rt_RGBDepth(localGradient, localHessian,
                                  locations, colours, rgb, depth,
                                  rgbImageSize, depthImageSize, locId_global,
                                  projRGBParams, projDepthParams, toRGBM, toDepthM,
                                  gx, gy, gdx, gdy, numPara, startPara);
    for (int i = 0; i < numParaSQ; ++i) 
      g_h_shared[locId_local * numParaSQ + i] = localHessian[i];
  } else {
    memset(g_h_shared + locId_local * numPara, 0, sizeof(float) * numParaSQ);
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (locId_local < offset) {
      for (int i = 0; i < numParaSQ; ++i)
        g_h_shared[locId_local * numParaSQ + i] += g_h_shared[(offset + locId_local) * numParaSQ + i];
    }
    __syncthreads();
  }

  if (locId_local == 0) {
    for (int i = 0; i < numParaSQ; ++i)
      h_device[blockIdx.x * numParaSQ + i] = g_h_shared[i];
  }
  __syncthreads();

  // Compute gradient
  if (locId_global < noTotalPoints) {
    for (int i = 0; i < numPara; ++i)
      g_h_shared[locId_local * numPara + i] = localGradient[i];
  } else {
    memset(g_h_shared + locId_local * numPara, 0, sizeof(float) * numPara);
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (locId_local < offset) {
      for (int i = 0; i < numPara; ++i)
        g_h_shared[locId_local * numPara + i] += g_h_shared[(offset + locId_local) * numPara + i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < numPara; ++i)
      g_device[blockIdx.x * numPara + i] = g_h_shared[i];
  }
}