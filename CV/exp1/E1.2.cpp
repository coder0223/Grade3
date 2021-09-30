#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

Mat init_image,bg_image,temp_image;
double res;
double thresh;//������ֵ

int main()
{
	init_image = imread("E:/1ComputerVision/exp/exp1/exp1.2/data/02.jpg");
	bg_image = imread("E:/1ComputerVision/exp/exp1/exp1.2/data/02_bg.jpg");//��ȡ����ͼƬ�ͱ���ͼƬ
	if (init_image.empty() || bg_image.empty())//��ͼƬ�������򷵻ر�����Ϣ���˳�
	{
		cout << "Image can not load!\n";
		return -1;
	}
	if (init_image.rows != bg_image.rows || init_image.cols != bg_image.cols)//����ͼƬ��ԭͼ�ߴ粻һ�޷����
	{
		cout << "The size are not the same, can not minus!\n";
		return -1;
	}
	temp_image = Mat::zeros(init_image.size(), init_image.type());//����ԭͼ
	namedWindow("Minus", WINDOW_AUTOSIZE);//��������
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
			res = sqrt(res);//��ԭͼ�ͱ���ÿ��ͨ���Ĳ�ֵ��ƽ���Ͳ�����
			if (res < thresh)//С���趨����ֵ��Ϊ��ɫ
			{
				for (int c = 0; c < 3; c++)
				{
					temp_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(0);
				}
			}
			else//������ֵ��Ϊ��ɫ����ǰ������
			{
				for (int c = 0; c < 3; c++)
				{
					temp_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(255);//��ʹ��saturate_cast()������ֹ��ɫ���
				}
			}
		}
	}
	imshow("Minus", temp_image);//��ʾ�趨�Ĵ���
	waitKey(0);
	return 0;
}