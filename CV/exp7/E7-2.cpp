#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

int main()
{
	double time0 = static_cast<double>(getTickCount());
	Mat img = imread("E:/1ComputerVision/exp/exp7/pic3.jpg");
	if (img.empty())
	{
		cout << "image read failed!" << endl;
		return -1;
	}
	Mat temp = img.clone();
	Mat gray;
	cvtColor(temp, gray, COLOR_RGB2GRAY);
	Mat dstImg, norImg, scaledImg;
	dstImg = Mat::zeros(temp.size(), CV_32FC1);
	cornerHarris(gray, dstImg, 2, 3, 0.04, BORDER_DEFAULT);
	normalize(dstImg, norImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	int Thresh = 100;
	for (int y = 0; y < norImg.rows; y++)
	{
		for (int x = 0; x < norImg.cols; x++)
		{
			if ((int)norImg.at<float>(y, x) > Thresh)
			{
				circle(temp, Point(x, y), 2, Scalar(10, 50, 255), 2, 8, 0);
			}
		}
	}	
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "runtime £º" << time0 << "s" << endl;
	imshow("image", temp);
	waitKey(0);
	return 0;
}