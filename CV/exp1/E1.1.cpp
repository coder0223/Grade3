#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;

int contr;//对比度变量
Mat image, temp;//图片

static void Contrast(int, void*)//调整对比度的函数
{
	namedWindow("Image", WINDOW_AUTOSIZE);//创建窗口
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)//设置每个像素点(x,y)每个通道c的值
			{
				double p = ((image.at<Vec3b>(y, x)[c] - 127) / 225.00) * contr * 0.1;//sigmoid函数原型为1/(1+e^(-x))
				temp.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(image.at<Vec3b>(y, x)[c] * ((1.00 / (1.00 + exp(-p))) + 0.3));//sigmiod函数
			}	//saturate_cast对图像色彩进行保护，防止像素溢出
		}
	}
	imshow("Image", image);//展示初始图片
	imshow("Slider", temp);//展示对比度可调的图片
}
int main()
{
	image = imread("E:/1ComputerVision/exp/exp1/exp1.1/pict.jpg");//读取图片
	if (image.empty())
	{
		cout << "image can not load!\n";
		return -1;
	}//若图片不存在，返回报错信息并退出
	temp = Mat::zeros(image.size(), image.type());//创建一张和原始图片完全一致的图片
	contr = 20;//初始对比度设置为20
	namedWindow("Slider", WINDOW_AUTOSIZE);//创建Slider滑动条窗口
	createTrackbar("Contrast", "Slider", &contr, 150, Contrast);//创建滑动条的函数
	//轨迹条名称、窗口名称、滑块初始位置、滑块能到达的最大值、指向回调函数
	Contrast(contr, 0);
	waitKey(0);
	return 0;
}