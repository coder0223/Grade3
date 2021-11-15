#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

const double PI = 3.1415926;

Mat addSalNoise(Mat img, int n)//增加椒盐噪声
{
	Mat res = img.clone();//克隆原图像
	//Mat res = Mat::zeros(img.size(), img.type());
	for (int k = 0; k < n; k++)
	{
		int i = rand() % res.cols;
		int j = rand() % res.rows;//随机取某一位置的像素
		for (int c = 0; c < 3; c++)
		{
			res.at<Vec3b>(j, i)[c] = 255;//将该位置的像素改为白色
		}
	}
	return res;
}

void Gaussian(const Mat& input, Mat& output, double sigma)
{
	int w_size = (int)(6 * sigma - 1);//滤波窗口大小
	if (w_size % 2 == 0)
	{
		cout << "the filter window size is invalid!";
		w_size -= 1;
	}
	int row = input.rows, col = input.cols;
	double* window = new double[w_size];//创建滤波窗口
	Mat newInput;
	copyMakeBorder(input, newInput, w_size / 2, w_size / 2, w_size / 2, w_size / 2, BORDER_REFLECT_101);
	//扩充边界，向四周各扩充w_size/2，采用BORDER_REFLECT_101方式，以边缘像素为轴对称填充

	double sum = 0;
	for (int i = 0; i < w_size; i++)
	{
		int center = w_size / 2;//滤波窗口中心
		window[i] = exp(-pow((i - center), 2) / (2 * sigma * sigma));//利用高斯函数求窗口内元素的值
		sum += window[i];
	}
	for (int i = 0; i < w_size; i++)
	{
		window[i] /= sum;//归一化
	}

	//对每一行进行一维高斯滤波
	for (int y = w_size / 2; y < row + w_size / 2; y++)
	{
		for (int x = w_size / 2; x < col + w_size / 2; x++)
		{
			int id = 0;
			double temp[3] = {0, 0, 0};
			for (int k = x - w_size / 2; k < x + w_size / 2; k++)
			{
				for (int c = 0; c < 3; c++)
				{
					temp[c] += (newInput.at<Vec3b>(y, k)[c]) * window[id];
				}
				id++;
			}
			for (int c = 0; c < 3; c++)
			{
				newInput.at<Vec3b>(y, x)[c] = temp[c];
			}
		}
	}

	//对每一列进行一维高斯滤波
	for (int x = w_size / 2; x < col + w_size / 2; x++)
	{
		for (int y = w_size / 2; y < row + w_size / 2; y++)
		{
			int id = 0;
			double temp[3] = { 0,0,0 };
			for (int k = y - w_size / 2; k < y + w_size / 2; k++)
			{
				for (int c = 0; c < 3; c++)
				{
					temp[c] += (newInput.at<Vec3b>(k, x)[c]) * window[id];
				}
				id++;
			}
			for (int c = 0; c < 3; c++)
			{
				newInput.at<Vec3b>(y, x)[c] = temp[c];
			}
		}
	}
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			output.at<Vec3b>(y, x) = newInput.at<Vec3b>(y + w_size / 2, x + w_size / 2);
		}
	}
}

int main()
{
	Mat img = imread("E:/1ComputerVision/exp/exp3/pic.jpg");
	if (img.empty())
	{
		cout << "Image Read Failed!";
		return -1;
	}
	//Mat ans = addSalNoise(img, 300);
	//namedWindow("image", WINDOW_AUTOSIZE);
	//cv::imshow("image", ans);
	//cv::imwrite("E:/1ComputerVision/exp/exp3/co.jpg",ans);
	int sigma = 3;
	imshow("image", img);
	Mat ans=Mat::zeros(img.size(),img.type());
	Gaussian(img, ans, sigma);
	imshow("dest", ans);
	//cv::imwrite("E:/1ComputerVision/exp/exp3/co.jpg", ans);
	waitKey(0);
	return 0;
}