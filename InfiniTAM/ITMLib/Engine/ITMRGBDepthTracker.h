#pragma once

#include "../Utils/ITMLibDefines.h"

#include "../Objects/ITMImageHierarchy.h"
#include "../Objects/ITMViewHierarchyLevel.h"

#include "../Engine/ITMTracker.h"
#include "../Engine/ITMLowLevelEngine.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
	namespace Engine
	{
		/** Base class for engines performing point based colour
		    tracking. Implementations would typically project down a
		    point cloud into observed images and try to minimize the
		    reprojection error.
		*/
		class ITMRGBDepthTracker : public ITMTracker
		{
		private:
			const ITMLowLevelEngine *lowLevelEngine;

			void PrepareForEvaluation(const ITMView *view);

		protected: 
			TrackerIterationType iterationType;
			ITMTrackingState *trackingState; const ITMView *view;
			ITMImageHierarchy<ITMViewHierarchyLevel> *viewHierarchy;
			int levelId;

			int countedPoints_valid;
		public:
			class EvaluationPoint
			{
			public:
				float f(void) { return cacheF; }
				const float* nabla_f(void) { if (cacheNabla == NULL) computeGradients(false); return cacheNabla; }

				const float* hessian_GN(void) { if (cacheHessian == NULL) computeGradients(true); return cacheHessian; }
				const ITMPose & getToRGBParameter(void) const { return *mToRGBPara; }
				const ITMPose & getToDepthParameter(void) const { return *mToDepthPara; }

				EvaluationPoint(ITMPose *toRGBPos, ITMPose *toDepthPos, const ITMRGBDepthTracker *f_parent);
				~EvaluationPoint(void)
				{
					delete mToRGBPara;
					delete mToDepthPara;
					if (cacheNabla != NULL) delete[] cacheNabla;
					if (cacheHessian != NULL) delete[] cacheHessian;
				}

			protected:
				void computeGradients(bool requiresHessian);

				ITMPose *mToRGBPara;
				ITMPose *mToDepthPara;
				const ITMRGBDepthTracker *mParent;

				float cacheF;
				float *cacheNabla;
				float *cacheHessian;
			};

			EvaluationPoint* evaluateAt(ITMPose *toRGBPara, ITMPose *toDepthPara) const
			{
				return new EvaluationPoint(toRGBPara, toDepthPara, this);
			}

			int numParameters(void) const { return (iterationType == TRACKER_ITERATION_ROTATION) ? 3 : 6; }

			virtual void F_oneLevel(float *f, ITMPose *pose) = 0;
			virtual void G_oneLevel(float *gradient, float *hessian, ITMPose *toRGBPose, ITMPose *toDepthPose) const = 0;

			void ApplyDelta(const ITMPose & para_old, const float *delta, ITMPose & para_new) const;

			void TrackCamera(ITMTrackingState *trackingState, const ITMView *view);

			ITMRGBDepthTracker(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels,
				const ITMLowLevelEngine *lowLevelEngine, MemoryDeviceType memoryType);
			virtual ~ITMRGBDepthTracker(void);
		};
	}
}
