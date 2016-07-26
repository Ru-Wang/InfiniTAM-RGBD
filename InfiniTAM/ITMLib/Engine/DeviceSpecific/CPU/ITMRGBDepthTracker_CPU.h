#pragma once

#include "../../ITMRGBDepthTracker.h"

namespace ITMLib
{
	namespace Engine
	{
		class ITMRGBDepthTracker_CPU : public ITMRGBDepthTracker
		{
		public:
			void F_oneLevel(float *f, ITMPose *toRGBPose);
			void G_oneLevel(float *gradient, float *hessian, ITMPose *toRGBPose, ITMPose *toDepthPose) const;

			ITMRGBDepthTracker_CPU(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels,
				const ITMLowLevelEngine *lowLevelEngine);
			~ITMRGBDepthTracker_CPU(void);
		};
	}
}
