#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("E:/1ComputerVision/exp/exp6/Circle Detection Dataset/60.jpg");
	if (img.empty())
	{
		cout << "image read failed!";
		return -1;
	}
	Mat ans = img.clone();
	cvtColor(img, img, CV_RGB2GRAY);
	imshow("image1", img);
	GaussianBlur(img, img, Size(9, 9), 2, 2);
	imshow("image2", img);
	vector<Vec3f> pc;
	Mat temp = img.clone();
	Canny(temp, temp, 200, 100, 3);
	//Canny()函数用于边缘检测
	imshow("image3", temp);
	HoughCircles(img, pc, CV_HOUGH_GRADIENT, 1, 20, 100, 100, 0, 0);
	//HoughCircles()霍夫圆变换函数
	for (size_t i = 0; i < pc.size(); ++i)
	{
		circle(ans, Point(pc[i][0], pc[i][1]), pc[i][2], Scalar(155, 50, 255), 3, 8, 0);
	}
	imshow("ans", ans);
	waitKey(0);
	return 0;
}