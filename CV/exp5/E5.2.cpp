#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("E:/1ComputerVision/exp/exp5/pic2.jpg");
	if (img.empty())
	{
		cout << "image read fail!";
		return -1;
	}
	imshow("image", img);
	Mat imgGray, imgBinary, dstImg;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);//转化为灰度图
	threshold(imgGray, imgBinary, 85, 255, THRESH_BINARY);//转化为二值图
	distanceTransform(imgBinary, dstImg, CV_DIST_L2, CV_DIST_MASK_PRECISE);//距离变换函数
	normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);//归一化
	imshow("ImgBinary", imgBinary);
	imshow("dstImg", dstImg);
	//cout << dstImg.channels();//dstImg是单通道图像
	waitKey(0);
	return 0;
}