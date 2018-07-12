#include <iostream>   
#include "opencv2/core/core.hpp"   
#include "opencv2/features2d/features2d.hpp"   
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/legacy/legacy.hpp"
#include <iostream>   
#include <vector>  
#include <time.h>
// GPU
#include "opencv2/gpu/gpu.hpp"
using namespace cv;
using namespace std;
using namespace cv::gpu;

int main()
{
	int num_devices = gpu::getCudaEnabledDeviceCount();
	cout << num_devices << endl;
	if (num_devices <= 0)
	{
		std::cerr << "There is no device." << std::endl;
		return - 1;
	}
	int enable_device_id = -1;
	for (int i = 0; i < num_devices; i++)
	{
		cv::gpu::DeviceInfo dev_info(i);
		if (dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	if (enable_device_id < 0)
	{
		std::cerr << "GPU module isn't built for GPU" << std::endl;
		return - 1;
	}
	gpu::setDevice(enable_device_id);

	IplImage* grayImg_1;
	IplImage* grayImg_2;
	Mat img_cpu_1 = imread("E:\\capture\\images5\\A005.mpg3700.jpg", 0);
	Mat img_cpu_2 = imread("E:\\capture\\images5\\A005.mpg3720.jpg", 0);
	//IplImage* Img_1 = cvLoadImage("E:\\capture\\images5\\A005.mpg3700.jpg");
	//IplImage* Img_2 = cvLoadImage("E:\\capture\\images5\\A005.mpg3701.jpg");
	//grayImg_1 = cvCreateImage(cvSize(Img_1->width, Img_1->height), 8, 1);
	//cvCvtColor(Img_1, grayImg_1, CV_BGR2GRAY);
	//grayImg_2 = cvCreateImage(cvSize(Img_2->width, Img_2->height), 8, 1);
	//cvCvtColor(Img_2, grayImg_2, CV_BGR2GRAY);

	GpuMat img_gpu_1(img_cpu_1);
	GpuMat img_gpu_2(img_cpu_2);

	GpuMat img_gray_gpu_1;
	GpuMat img_gray_gpu_2;
	//gpu::createContinuous(img_gpu_1.rows, );
	//gpu::cvtColor(img_gpu_1, img_gray_gpu_1, CV_BGR2GRAY);
	//gpu::cvtColor(img_gpu_2, img_gray_gpu_2, CV_BGR2GRAY);

	ORB_GPU orb_gpu;

	clock_t start, end;
	vector<KeyPoint> dkeyPoints_1, dkeyPoints_2;
	GpuMat ddescriptors_1, ddescriptors_2;
	Mat des_1, des_2;
	
	start = clock();
	orb_gpu(img_gpu_1, GpuMat(), dkeyPoints_1, ddescriptors_1);
	orb_gpu(img_gpu_2, GpuMat(), dkeyPoints_2, ddescriptors_2);
	end = clock();
	cout << " orb_gpu (1300*900)耗时："<<(double)(end - start)/2 <<" ms" <<endl;
	//cout << "FOUND " << dkeyPoints_1.cols << " keypoints on first image" << endl;
	//cout << "FOUND " << dkeyPoints_2.cols << " keypoints on second image" << endl;

	Mat img_cpu_1_xx = imread("E:\\capture\\images5\\A005.mpg3700.jpg");
	Mat img_cpu_2_xx = imread("E:\\capture\\images5\\A005.mpg3710.jpg");
	GpuMat img_gpu_1_xx(img_cpu_1_xx);
	GpuMat img_gpu_2_xx(img_cpu_2_xx);

	GpuMat img_gray_gpu_1_xx;
	GpuMat img_gray_gpu_2_xx;
	//gpu::createContinuous(img_gpu_1.rows, );
	gpu::cvtColor(img_gpu_1_xx, img_gray_gpu_1_xx, CV_BGR2GRAY);
	gpu::cvtColor(img_gpu_2_xx, img_gray_gpu_2_xx, CV_BGR2GRAY);
	//
	ddescriptors_1.download(des_1);
	ddescriptors_2.download(des_2);

	Mat img_gray_cpu_1_xx;
	Mat img_gray_cpu_2_xx;
	img_gray_gpu_1_xx.download(img_gray_cpu_1_xx);
	img_gray_gpu_2_xx.download(img_gray_cpu_2_xx);
	imwrite("E:\\capture\\images5\\orb\\des_1.bmp", des_1);
	imwrite("E:\\capture\\images5\\orb\\des_2.bmp", des_2);
	imwrite("E:\\capture\\images5\\orb\\img_gray_cpu_1_xx.bmp", img_gray_cpu_1_xx);
	imwrite("E:\\capture\\images5\\orb\\img_gray_cpu_2_xx.bmp", img_gray_cpu_2_xx);

	BruteForceMatcher_GPU<Hamming> matcher;
	vector<DMatch> matches;
	//const GpuMat mask;
	matcher.match(ddescriptors_1, ddescriptors_2, matches, GpuMat());
	double max_dist;
	double min_dist;
	//-- Quick calculation of max and min distances between keypoints     
	for (int i = 0; i < des_1.rows; i++)
	{
		double dist = matches[i].distance;

		if (i == 0)
		{
			min_dist = dist;
			max_dist = dist;
		}

		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )     
	//-- PS.- radiusMatch can also be used here.     
	vector< DMatch > good_matches;
	for (int i = 0; i < des_1.rows; i++)
	{
		if (matches[i].distance < 0.50*max_dist )
		{
			good_matches.push_back(matches[i]);
		}
	}
	Mat img_matches;
	drawMatches(img_cpu_1, dkeyPoints_1, img_cpu_2, dkeyPoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	cout << good_matches.size() << endl;
	for (size_t i = 0; i < good_matches.size(); ++i)
	{
		// get the keypoints from the good matches
		obj.push_back(dkeyPoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(dkeyPoints_2[good_matches[i].trainIdx].pt);
	}
	const CvMat H = findHomography(obj, scene, CV_RANSAC);
	const Mat H_p = findHomography(obj, scene, CV_RANSAC);
	CvMat H_pp = findHomography(obj, scene, CV_RANSAC);
	//imwrite("E:\\capture\\images5\\orb\\H.bmp", H);
	imwrite("E:\\capture\\images5\\orb\\H_p.bmp", H_p);
	//imwrite("E:\\capture\\images5\\orb\\H_pp.bmp", H_pp);
	
	//test
	if (1)
	{
		// get the corners from the image_1
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_cpu_1.cols, 0);
		obj_corners[2] = cvPoint(img_cpu_1.cols, img_cpu_1.rows);
		obj_corners[3] = cvPoint(0, img_cpu_1.rows);
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H_p);

		// draw lines between the corners (the mapped object in the scene - image_2)
		line(img_matches, scene_corners[0] + Point2f(img_cpu_1.cols, 0), scene_corners[1] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
		line(img_matches, scene_corners[1] + Point2f(img_cpu_1.cols, 0), scene_corners[2] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
		line(img_matches, scene_corners[2] + Point2f(img_cpu_1.cols, 0), scene_corners[3] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
		line(img_matches, scene_corners[3] + Point2f(img_cpu_1.cols, 0), scene_corners[0] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));

	}

	imwrite("E:\\capture\\images5\\orb\\img_matches.bmp", img_matches);

	//cvSaveImageM("E:\\capture\\images5\\ddd.bmp", img_1, 0);

	//IplImage* xformed;
	//IplImage* imag;
	//IplImage* save;
	GpuMat xformed;
	GpuMat save;
	GpuMat img;
	if(NULL != 1)
	{
		//time_start = clock();
		//xformed = cvCreateImage(cvGetSize(grayImg_2), 8, 1);
		//imag = cvCreateImage(cvGetSize(grayImg_2), 8, 1);
		//save = cvCreateImage(cvGetSize(grayImg_2), 8, 1);
		//cv::Size size = img_gpu_1.size;

		gpu::warpPerspective(img_gpu_1, xformed, H_p, img_gpu_1.size(), INTER_LINEAR, BORDER_CONSTANT, cvScalarAll(0), Stream::Null());
		//cvWarpPerspective(grayImg_1, xformed, &H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));//放射变化 把图像img1映射在xformed上没有的地方添0
		//cvThreshold(xformed, imag, 0, 255, 0);

		Mat save_cpu;
		Mat img_cpu;
		xformed.download(save_cpu);

		imwrite("E:\\capture\\images5\\orb\\xformed.bmp", save_cpu);
		gpu::threshold(xformed, img, 0, 255, 0);
		gpu::subtract(img_gpu_2, xformed, save,GpuMat(), 0);
		
		save.download(save_cpu);
		img.download(img_cpu);

		IplImage ipl_save_d = save_cpu;
		IplImage ipl_img_d = img_cpu;
		
		IplImage* ipl_save = cvCloneImage(&ipl_save_d);
		IplImage* ipl_img = cvCloneImage(&ipl_img_d);

		int x, y, c;
		for (y = 0; y<ipl_save -> height; y++)
			for (x = 0; x< ipl_save->width; x++)
			{
				c = ((uchar*)(ipl_img->imageData + ipl_img->widthStep*y))[x];
				if (c == 0)
					((uchar*)(ipl_save->imageData + ipl_save->widthStep*y))[x] = (uchar)c;
			}

		cvSaveImage("E:\\capture\\images5\\orb\\ipl_save.bmp", ipl_save);
		//cvSaveImage("E:\\capture\\images5\\orb\\save.bmp", ipl_save, 0);
		//imwrite("E:\\capture\\images5\\orb\\save.bmp", ipl_save);

		/*
		int x;
		int y;
		int c;
		for (y = 0; y<save->height; y++)
			for (x = 0; x<save->width; x++)
			{
				c = ((uchar*)(imag->imageData + imag->widthStep*y))[x];
				if (c == 0)
					((uchar*)(save->imageData + save->widthStep*y))[x] = (uchar)c;
			}

		cvSaveImage("E:\\capture\\images5\\orb\\grayImg_1.bmp", grayImg_1);
		cvSaveImage("E:\\capture\\images5\\orb\\grayImg_2.bmp", grayImg_2);
		cvSaveImage("E:\\capture\\images5\\orb\\save.bmp", save);
		*/

	}

	return 0;
}









