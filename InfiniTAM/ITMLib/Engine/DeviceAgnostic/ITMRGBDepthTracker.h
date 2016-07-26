#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"

_CPU_AND_GPU_CODE_ inline float getColorDifferenceSq(
    const CONSTPTR(Vector4f) *locations,
    const CONSTPTR(Vector4f) *colours,
    const CONSTPTR(Vector4u) *rgb,
    const CONSTPTR(Vector2i) & rgbImageSize,
    int locId_global, Vector4f projParams, Matrix4f M)
{
	Vector4f pt_model, pt_camera, colour_known, colour_obs;
	Vector3f colour_diff;
	Vector2f pt_image;

	pt_model = locations[locId_global];
	colour_known = colours[locId_global];

	pt_camera = M * pt_model;

	if (pt_camera.z <= 0) return -1.0f;

	pt_image.x = projParams.x * pt_camera.x / pt_camera.z + projParams.z;
	pt_image.y = projParams.y * pt_camera.y / pt_camera.z + projParams.w;

	if (pt_image.x < 0 || pt_image.x > rgbImageSize.x - 1 || pt_image.y < 0 || pt_image.y > rgbImageSize.y - 1) return -1.0f;

	colour_obs = interpolateBilinear(rgb, pt_image, rgbImageSize);
	if (colour_obs.w < 254.0f) return -1.0f;

	colour_diff.x = colour_obs.x - 255.0f * colour_known.x;
	colour_diff.y = colour_obs.y - 255.0f * colour_known.y;
	colour_diff.z = colour_obs.z - 255.0f * colour_known.z;

	return colour_diff.x * colour_diff.x + colour_diff.y * colour_diff.y + colour_diff.z * colour_diff.z;
}

_CPU_AND_GPU_CODE_ inline bool computePerPointGH_rt_RGBDepth(
    THREADPTR(float) *localGradient,
    THREADPTR(float) *localHessian,
    const CONSTPTR(Vector4f) *locations,
    const CONSTPTR(Vector4f) *colours,
    const CONSTPTR(Vector4u) *rgb,
    const CONSTPTR(float) 	 *depth,
    const CONSTPTR(Vector2i) & rgbImageSize,
    const CONSTPTR(Vector2i) & depthImageSize,
    int locId_global,
    const CONSTPTR(Vector4f) & projRGBParams,
    const CONSTPTR(Vector4f) & projDepthParams,
    const CONSTPTR(Matrix4f) & toRGBM,
    const CONSTPTR(Matrix4f) & toDepthM,
    const CONSTPTR(Vector4s) *gx, const CONSTPTR(Vector4s) *gy,
    const CONSTPTR(float) 	 *gdx, const CONSTPTR(float) 	 *gdy,
    int numPara, int startPara)
{
	Vector4f pt_model, pt_rgb_camera, pt_depth_camera, colour_known, colour_obs, gx_obs, gy_obs;
	Vector4f rgbd_diff_d, d[6];
	Vector3f d_pt_cam_dpi;
	Vector2f pt_image, pt_depth, d_proj_dpi;
	float depth_known, depth_obs, gdx_obs, gdy_obs;

	pt_model = locations[locId_global];
	colour_known = colours[locId_global];

	pt_rgb_camera = toRGBM * pt_model;
	pt_depth_camera = toDepthM * pt_model;

	if (pt_rgb_camera.z <= 0 || pt_depth_camera.z <= 0) return false;

	depth_known = 1 / pt_depth_camera.z;

	pt_image.x = projRGBParams.x * pt_rgb_camera.x / pt_rgb_camera.z + projRGBParams.z;
	pt_image.y = projRGBParams.y * pt_rgb_camera.y / pt_rgb_camera.z + projRGBParams.w;

	pt_depth.x = projDepthParams.x * pt_depth_camera.x / pt_depth_camera.z + projDepthParams.z;
	pt_depth.y = projDepthParams.y * pt_depth_camera.y / pt_depth_camera.z + projDepthParams.w;

	if (pt_image.x < 0 || pt_image.x > rgbImageSize.x - 1 || pt_image.y < 0 || pt_image.y > rgbImageSize.y - 1) return false;
	if (pt_depth.x < 0 || pt_depth.x > depthImageSize.x - 1 || pt_depth.y < 0 || pt_depth.y > depthImageSize.y - 1) return false;

	colour_obs = interpolateBilinear(rgb, pt_image, rgbImageSize);
	gx_obs = interpolateBilinear(gx, pt_image, rgbImageSize);
	gy_obs = interpolateBilinear(gy, pt_image, rgbImageSize);

	depth_obs = interpolateBilinear_withHoles_single(depth, pt_depth, depthImageSize);
	gdx_obs = interpolateBilinear_withHoles_single(gdx, pt_depth, depthImageSize);
	gdy_obs = interpolateBilinear_withHoles_single(gdy, pt_depth, depthImageSize);

	if (colour_obs.w < 254.0f) return false;
	if (depth_obs <= 0.0) return false;

	depth_obs = 1 / depth_obs;

	rgbd_diff_d.x = 2.0f * (colour_obs.x - 255.0f * colour_known.x);
	rgbd_diff_d.y = 2.0f * (colour_obs.y - 255.0f * colour_known.y);
	rgbd_diff_d.z = 2.0f * (colour_obs.z - 255.0f * colour_known.z);
	rgbd_diff_d.w = 2.0f * (depth_obs - depth_known);

	for (int para = 0, counter = 0; para < numPara; para++)
	{
		switch (para + startPara)
		{
		case 0: d_pt_cam_dpi.x = pt_rgb_camera.w;  d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = 0.0f;         break;
		case 1: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = pt_rgb_camera.w;  d_pt_cam_dpi.z = 0.0f;         break;
		case 2: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = pt_rgb_camera.w;  break;
		case 3: d_pt_cam_dpi.x = 0.0f;         d_pt_cam_dpi.y = -pt_rgb_camera.z;  d_pt_cam_dpi.z = pt_rgb_camera.y;  break;
		case 4: d_pt_cam_dpi.x = pt_rgb_camera.z;  d_pt_cam_dpi.y = 0.0f;         d_pt_cam_dpi.z = -pt_rgb_camera.x;  break;
		default:
		case 5: d_pt_cam_dpi.x = -pt_rgb_camera.y;  d_pt_cam_dpi.y = pt_rgb_camera.x;  d_pt_cam_dpi.z = 0.0f;         break;
		};

		d_proj_dpi.x = projRGBParams.x * ((pt_rgb_camera.z * d_pt_cam_dpi.x - d_pt_cam_dpi.z * pt_rgb_camera.x) / (pt_rgb_camera.z * pt_rgb_camera.z));
		d_proj_dpi.y = projRGBParams.y * ((pt_rgb_camera.z * d_pt_cam_dpi.y - d_pt_cam_dpi.z * pt_rgb_camera.y) / (pt_rgb_camera.z * pt_rgb_camera.z));

		d[para].x = d_proj_dpi.x * gx_obs.x + d_proj_dpi.y * gy_obs.x;
		d[para].y = d_proj_dpi.x * gx_obs.y + d_proj_dpi.y * gy_obs.y;
		d[para].z = d_proj_dpi.x * gx_obs.z + d_proj_dpi.y * gy_obs.z;
		d[para].w = d_proj_dpi.x * gdx_obs  + d_proj_dpi.y * gdy_obs;

		localGradient[para] = d[para].x * rgbd_diff_d.x + d[para].y * rgbd_diff_d.y + d[para].z * rgbd_diff_d.z + d[para].w * rgbd_diff_d.w;

		for (int col = 0; col <= para; col++)
			localHessian[counter++] = 2.0f * (d[para].x * d[col].x + d[para].y * d[col].y + d[para].z * d[col].z + d[para].w * d[para].w);
	}

	return true;
}
