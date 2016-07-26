#include "ITMRGBDepthTracker.h"
#include "../../ORUtils/Cholesky.h"

#include <math.h>

using namespace ITMLib::Engine;

static inline bool minimizeLM(const ITMRGBDepthTracker & tracker, ITMPose & initToRGB, ITMPose & initToDepth);

ITMRGBDepthTracker::ITMRGBDepthTracker(Vector2i imgSize, TrackerIterationType *trackingRegime, int noHierarchyLevels,
	const ITMLowLevelEngine *lowLevelEngine, MemoryDeviceType memoryType)
{
	viewHierarchy = new ITMImageHierarchy<ITMViewHierarchyLevel>(imgSize, trackingRegime, noHierarchyLevels, memoryType);

	this->lowLevelEngine = lowLevelEngine;
}

ITMRGBDepthTracker::~ITMRGBDepthTracker(void)
{
	delete viewHierarchy;
}

void ITMRGBDepthTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
{
	this->view = view; this->trackingState = trackingState;

	this->PrepareForEvaluation(view);

	ITMPose toDepthCameraPara(trackingState->pose_d->GetM());
	ITMPose toRGBCameraPara(view->calib->trafo_rgb_to_depth.calib_inv * trackingState->pose_d->GetM());
	for (int levelId = viewHierarchy->noLevels - 1; levelId >= 0; levelId--)
	{
		this->levelId = levelId;
		this->iterationType = viewHierarchy->levels[levelId]->iterationType;

		minimizeLM(*this, toRGBCameraPara, toDepthCameraPara);
	}

	// these following will coerce the result back into the chosen
	// parameterization for rotations
	trackingState->pose_d->SetM(view->calib->trafo_rgb_to_depth.calib * toRGBCameraPara.GetM());

	trackingState->pose_d->Coerce();

	//printf(">> %f %f %f %f %f %f\n", scene->pose->params.each.rx, scene->pose->params.each.ry, scene->pose->params.each.rz,
	//	scene->pose->params.each.tx, scene->pose->params.each.ty, scene->pose->params.each.tz);
}

void ITMRGBDepthTracker::PrepareForEvaluation(const ITMView *view)
{
	viewHierarchy->levels[0]->rgb->ChangeDims(view->rgb->noDims);
	viewHierarchy->levels[0]->depth->ChangeDims(view->depth->noDims);
	lowLevelEngine->CopyImage(viewHierarchy->levels[0]->rgb, view->rgb);
	lowLevelEngine->CopyImage(viewHierarchy->levels[0]->depth, view->depth);

	ITMImageHierarchy<ITMViewHierarchyLevel> *hierarchy = viewHierarchy;

	for (int i = 1; i < hierarchy->noLevels; i++)
	{
		ITMViewHierarchyLevel *currentLevel = hierarchy->levels[i], *previousLevel = hierarchy->levels[i - 1];
		lowLevelEngine->FilterSubsample(currentLevel->rgb, previousLevel->rgb);
		lowLevelEngine->FilterSubsampleWithHoles(currentLevel->depth, previousLevel->depth);
	}

	for (int i = 0; i < hierarchy->noLevels; i++)
	{
		ITMViewHierarchyLevel *currentLevel = hierarchy->levels[i];

		lowLevelEngine->GradientX(currentLevel->gradientX_rgb, currentLevel->rgb);
		lowLevelEngine->GradientY(currentLevel->gradientY_rgb, currentLevel->rgb);
		lowLevelEngine->GradientX(currentLevel->gradientX_depth, currentLevel->depth);
		lowLevelEngine->GradientY(currentLevel->gradientY_depth, currentLevel->depth);
	}
}

void ITMRGBDepthTracker::ApplyDelta(const ITMPose & para_old, const float *delta, ITMPose & para_new) const
{
	float paramVector[6];

	switch (iterationType)
	{
	case TRACKER_ITERATION_ROTATION:
		paramVector[0] = 0.0f; paramVector[1] = 0.0f; paramVector[2] = 0.0f;
		paramVector[3] = (float)(delta[0]); paramVector[4] = (float)(delta[1]); paramVector[5] = (float)(delta[2]);
		break;
	case TRACKER_ITERATION_TRANSLATION:
		paramVector[0] = (float)(delta[0]); paramVector[1] = (float)(delta[1]); paramVector[2] = (float)(delta[2]);
		paramVector[3] = 0.0f; paramVector[4] = 0.0f; paramVector[5] = 0.0f;
		break;
	case TRACKER_ITERATION_BOTH:
		paramVector[0] = (float)(delta[0]); paramVector[1] = (float)(delta[1]); paramVector[2] = (float)(delta[2]);
		paramVector[3] = (float)(delta[3]); paramVector[4] = (float)(delta[4]); paramVector[5] = (float)(delta[5]);
		break;
	default: break;
	}

	para_new.SetFrom(paramVector);
	para_new.MultiplyWith(&(para_old));
}

void ITMRGBDepthTracker::EvaluationPoint::computeGradients(bool hessianRequired)
{
	int numPara = mParent->numParameters();
	cacheNabla = new float[numPara];
	cacheHessian = new float[numPara*numPara];

	mParent->G_oneLevel(cacheNabla, cacheHessian, mToRGBPara, mToDepthPara);
}

ITMRGBDepthTracker::EvaluationPoint::EvaluationPoint(ITMPose *toRGBPos, ITMPose *toDepthPose, const ITMRGBDepthTracker *f_parent)
{
	float localF[1];

	this->mToRGBPara = toRGBPos; this->mToDepthPara = toDepthPose; this->mParent = f_parent;

	ITMRGBDepthTracker *parent = (ITMRGBDepthTracker *)mParent;

	parent->F_oneLevel(localF, toRGBPos);

	cacheF = localF[0];

	cacheHessian = NULL; cacheNabla = NULL;
}

// LM optimisation

static inline double stepQuality(ITMRGBDepthTracker::EvaluationPoint *x, ITMRGBDepthTracker::EvaluationPoint *x2, const float *step, const float *grad, const float *B, int numPara)
{
	double actual_reduction = x->f() - x2->f();
	double predicted_reduction = 0.0;
	float *tmp = new float[numPara];

	matmul(B, step, tmp, numPara, numPara);
	for (int i = 0; i < numPara; i++) predicted_reduction -= grad[i] * step[i] + 0.5*step[i] * tmp[i];
	delete[] tmp;

	if (predicted_reduction < 0) return actual_reduction / fabs(predicted_reduction);
	return actual_reduction / predicted_reduction;
}

static inline bool minimizeLM(const ITMRGBDepthTracker & tracker, ITMPose & initToRGB, ITMPose & initToDepth)
{
	// These are some sensible default parameters for Levenberg Marquardt.
	// The first three control the convergence criteria, the others might
	// impact convergence speed.
	static const int MAX_STEPS = 100;
	static const float MIN_STEP = 0.00005f;
	static const float MIN_DECREASE = 0.00001f;
	static const float TR_QUALITY_GAMMA1 = 0.75f;
	static const float TR_QUALITY_GAMMA2 = 0.25f;
	static const float TR_REGION_INCREASE = 2.0f;
	static const float TR_REGION_DECREASE = 0.25f;

	int numPara = tracker.numParameters();
	float *d = new float[numPara];
	float lambda = 0.01f;
	int step_counter = 0;

	ITMRGBDepthTracker::EvaluationPoint *x = tracker.evaluateAt(new ITMPose(initToRGB), new ITMPose(initToDepth));
	ITMRGBDepthTracker::EvaluationPoint *x2 = NULL;

	if (!portable_finite(x->f())) { delete[] d; delete x; return false; }

	do
	{
		const float *grad;
		const float *B;

		grad = x->nabla_f();
		B = x->hessian_GN();

		bool success;
		{
			float *A = new float[numPara*numPara];
			for (int i = 0; i < numPara*numPara; ++i) A[i] = B[i];
			for (int i = 0; i < numPara; ++i)
			{
				float & ele = A[i*(numPara + 1)];
				if (!(fabs(ele) < 1e-15f)) ele *= (1.0f + lambda); else ele = lambda*1e-10f;
			}

			ORUtils::Cholesky cholA(A, numPara);

			cholA.Backsub(&(d[0]), grad);
			// TODO: if Cholesky failed, set success to false!

			success = true;
			delete[] A;
		}

		if (success)
		{
			float MAXnorm = 0.0;
			for (int i = 0; i<numPara; i++) { float tmp = fabs(d[i]); if (tmp>MAXnorm) MAXnorm = tmp; }

			if (MAXnorm < MIN_STEP) break;
			for (int i = 0; i < numPara; i++) d[i] = -d[i];

			// make step
			ITMPose *tmp_to_rgb_para = new ITMPose(x->getToRGBParameter());
			tracker.ApplyDelta(x->getToRGBParameter(), &(d[0]), *tmp_to_rgb_para);
			ITMPose *tmp_to_depth_para = new ITMPose(x->getToDepthParameter());
			tracker.ApplyDelta(x->getToDepthParameter(), &(d[0]), *tmp_to_depth_para);

			// check whether step reduces error function and
			// compute a new value of lambda
			x2 = tracker.evaluateAt(tmp_to_rgb_para, tmp_to_depth_para);

			double rho = stepQuality(x, x2, &(d[0]), grad, B, numPara);
			if (rho > TR_QUALITY_GAMMA1) lambda = lambda / TR_REGION_INCREASE;
			else if (rho <= TR_QUALITY_GAMMA2) { success = false; lambda = lambda / TR_REGION_DECREASE; }
		}
		else
		{
			x2 = NULL;
			// can't compute a step quality here...
			lambda = lambda / TR_REGION_DECREASE;
		}

		if (success)
		{
			// accept step
			bool continueIteration = true;
			if (!(x2->f() < (x->f() - fabs(x->f()) * MIN_DECREASE))) continueIteration = false;

			delete x;
			x = x2;

			if (!continueIteration) break;
		}
		else if (x2 != NULL) delete x2;
		if (step_counter++ >= MAX_STEPS - 1) break;
	} while (true);

	initToRGB.SetFrom(&(x->getToRGBParameter()));
	delete x;

	delete[] d;

	return true;
}
