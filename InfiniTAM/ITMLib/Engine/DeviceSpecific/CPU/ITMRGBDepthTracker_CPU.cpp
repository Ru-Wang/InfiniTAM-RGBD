#include "ITMRGBDepthTracker_CPU.h"
#include "../../DeviceAgnostic/ITMRGBDepthTracker.h"
#include "../../DeviceAgnostic/ITMPixelUtils.h"

using namespace ITMLib::Engine;

ITMRGBDepthTracker_CPU::ITMRGBDepthTracker_CPU(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels, const ITMLowLevelEngine *lowLevelEngine)
	: ITMRGBDepthTracker(imgSize, trackingRegime, noHierarchyLevels, lowLevelEngine, MEMORYDEVICE_CPU) {  }

ITMRGBDepthTracker_CPU::~ITMRGBDepthTracker_CPU(void) { }

void ITMRGBDepthTracker_CPU::F_oneLevel(float *f, ITMPose *toRGBPose)
{
	int noTotalPoints = trackingState->pointCloud->noTotalPoints;

	Vector4f projParams = view->calib->intrinsics_rgb.projectionParamsSimple.all;
	projParams.x /= 1 << levelId; projParams.y /= 1 << levelId;
	projParams.z /= 1 << levelId; projParams.w /= 1 << levelId;

	Matrix4f M = toRGBPose->GetM();

	Vector2i rgbImageSize = viewHierarchy->levels[levelId]->rgb->noDims;

	float scaleForOcclusions, final_f;

	Vector4f *locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CPU);
	Vector4f *colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = viewHierarchy->levels[levelId]->rgb->GetData(MEMORYDEVICE_CPU);

	final_f = 0; countedPoints_valid = 0;
	for (int locId = 0; locId < noTotalPoints; locId++)
	{
		float colorDiffSq = getColorDifferenceSq(locations, colours, rgb, rgbImageSize, locId, projParams, M);
		if (colorDiffSq >= 0) { final_f += colorDiffSq; countedPoints_valid++; }
	}

	if (countedPoints_valid == 0) { final_f = MY_INF; scaleForOcclusions = 1.0; }
	else { scaleForOcclusions = (float)noTotalPoints / countedPoints_valid; }

	f[0] = final_f * scaleForOcclusions;
}

void ITMRGBDepthTracker_CPU::G_oneLevel(float *gradient, float *hessian, ITMPose *toRGBPose, ITMPose *toDepthPose) const
{
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

	float scaleForOcclusions;

	bool rotationOnly = iterationType == TRACKER_ITERATION_ROTATION;
	int numPara = rotationOnly ? 3 : 6, startPara = rotationOnly ? 3 : 0, numParaSQ = rotationOnly ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;

	float globalGradient[6], globalHessian[21];
	for (int i = 0; i < numPara; i++) globalGradient[i] = 0.0f;
	for (int i = 0; i < numParaSQ; i++) globalHessian[i] = 0.0f;

	Vector4f *locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CPU);
	Vector4f *colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = viewHierarchy->levels[levelId]->rgb->GetData(MEMORYDEVICE_CPU);
	Vector4s *gx = viewHierarchy->levels[levelId]->gradientX_rgb->GetData(MEMORYDEVICE_CPU);
	Vector4s *gy = viewHierarchy->levels[levelId]->gradientY_rgb->GetData(MEMORYDEVICE_CPU);
	float *depth = viewHierarchy->levels[levelId]->depth->GetData(MEMORYDEVICE_CPU);
	float *gdx = viewHierarchy->levels[levelId]->gradientX_depth->GetData(MEMORYDEVICE_CPU);
	float *gdy = viewHierarchy->levels[levelId]->gradientY_depth->GetData(MEMORYDEVICE_CPU);

	for (int locId = 0; locId < noTotalPoints; locId++)
	{
		float localGradient[6], localHessian[21];

		bool isValidPoint = computePerPointGH_rt_RGBDepth(
        localGradient, localHessian,
        locations, colours, rgb, depth,
        rgbImageSize, depthImageSize, locId,
        projRGBParams, projDepthParams, toRGBM, toDepthM,
        gx, gy, gdx, gdy, numPara, startPara
		);

		if (isValidPoint)
		{
			for (int i = 0; i < numPara; i++) globalGradient[i] += localGradient[i];
			for (int i = 0; i < numParaSQ; i++) globalHessian[i] += localHessian[i];
		}
	}

	scaleForOcclusions = (float)noTotalPoints / countedPoints_valid;
	if (countedPoints_valid == 0) { scaleForOcclusions = 1.0f; }

	for (int para = 0, counter = 0; para < numPara; para++)
	{
		gradient[para] = globalGradient[para] * scaleForOcclusions;
		for (int col = 0; col <= para; col++, counter++) hessian[para + col * numPara] = globalHessian[counter] * scaleForOcclusions;
	}
	for (int row = 0; row < numPara; row++)
	{
		for (int col = row + 1; col < numPara; col++) hessian[row + col * numPara] = hessian[col + row * numPara];
	}
}

