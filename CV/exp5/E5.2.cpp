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
	cvtColor(img, imgGray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(imgGray, imgBinary, 85, 255, THRESH_BINARY);//ת��Ϊ��ֵͼ
	distanceTransform(imgBinary, dstImg, CV_DIST_L2, CV_DIST_MASK_PRECISE);//����任����
	normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);//��һ��
	imshow("ImgBinary", imgBinary);
	imshow("dstImg", dstImg);
	//cout << dstImg.channels();//dstImg�ǵ�ͨ��ͼ��
	waitKey(0);
	return 0;
}