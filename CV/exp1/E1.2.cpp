#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

Mat init_image,bg_image,temp_image;
double res;
double thresh;//设置阈值

int main()
{
	init_image = imread("E:/1ComputerVision/exp/exp1/exp1.2/data/02.jpg");
	bg_image = imread("E:/1ComputerVision/exp/exp1/exp1.2/data/02_bg.jpg");//读取完整图片和背景图片
	if (init_image.empty() || bg_image.empty())//若图片不存在则返回报错信息并退出
	{
		cout << "Image can not load!\n";
		return -1;
	}
	if (init_image.rows != bg_image.rows || init_image.cols != bg_image.cols)//背景图片与原图尺寸不一无法相减
	{
		cout << "The size are not the same, can not minus!\n";
		return -1;
	}
	temp_image = Mat::zeros(init_image.size(), init_image.type());//复制原图
	namedWindow("Minus", WINDOW_AUTOSIZE);//创建窗口
	thresh = 125;
	for (int y = 0; y < init_image.rows; y++)
	{
		for (int x = 0; x < init_image.cols; x++)
		{
			//for (int c = 0; c < 3; c++)
			//{
			//	temp_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(init_image.at<Vec3b>(y, x)[c] - bg_image.at<Vec3b>(y,x)[c]);
			//}
			res = 0.0;
			for (int c = 0; c < 3; c++)
			{
				res += pow((init_image.at<Vec3b>(y, x)[c] - bg_image.at<Vec3b>(y, x)[c]), 2);			
				//res += init_image.at<Vec3b>(y, x)[c] - bg_image.at<Vec3b>(y, x)[c];
			}
			res = sqrt(res);//对原图和背景每个通道的差值求平方和并开根
			if (res < thresh)//小于设定的阈值则为黑色
			{
				for (int c = 0; c < 3; c++)
				{
					temp_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(0);
				}
			}
			else//大于阈值则为白色，即前景区域
			{
				for (int c = 0; c < 3; c++)
				{
					temp_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(255);//均使用saturate_cast()函数防止颜色溢出
				}
			}
		}
	}
	imshow("Minus", temp_image);//显示设定的窗口
	waitKey(0);
	return 0;
}