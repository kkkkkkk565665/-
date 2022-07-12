#include"steger.h"
vector<Point2d> steger(Mat img0)
{
    vector<Point2d> POINT;
    Mat img = img0.clone();
    Mat img_threshold = img0.clone();
    img.convertTo(img, CV_64FC1);
    //img_threshold.convertTo(img_threshold, CV_8UC1);
    GaussianBlur(img, img, Size(3, 3), 3, 3);
    threshold(img_threshold, img_threshold, 50, 255, THRESH_BINARY);//使用大津法自定义阈值二值化图像
    //namedWindow("Example1",WINDOW_FREERATIO);
    //imshow("Example1",img_threshold);
    //waitKey(100);
    double X = 0.0;
    double Y = 0.0;
    double Sigma = 12.0 / sqrt(3);
    double pi = 3.1415926;
    Mat kernaldx = Mat::zeros(69, 69, CV_64FC1);
    Mat kernaldxx = kernaldx.clone();
    Mat kernaldxy = kernaldx.clone();
    for (int i = 0; i < kernaldx.rows; i++)
    {
        for (int j = 0; j < kernaldx.cols; j++)
        {
            X = double(i) - 34.0;
            Y = double(j) - 34.0;
            kernaldx.at<double>(i, j) = (1.0 / (2.0 * pi * pow(Sigma, 4))) * ((-1) * X) * (exp((-1) * ((pow(X, 2) + pow(Y, 2)) / (2 * pow(Sigma, 2)))));
            kernaldxx.at<double>(i, j) = (1.0 / (2.0 * pi * pow(Sigma, 4))) * ((pow(X, 2) / pow(Sigma, 2)) - 1) * (exp((-1) * ((pow(X, 2) + pow(Y, 2)) / (2 * pow(Sigma, 2)))));
            kernaldxy.at<double>(i, j) = (1.0 / (2.0 * pi * pow(Sigma, 6))) * (Y * X) * (exp((-1) * ((pow(X, 2) + pow(Y, 2)) / (2 * pow(Sigma, 2)))));
        }
    }
    Mat dx, dy;
    Mat dxx, dyy, dxy;
    filter2D(img, dx, CV_64F, kernaldx, Point(-1, -1), 0, BORDER_CONSTANT);
    filter2D(img, dy, CV_64F, kernaldx.t(), Point(-1, -1), 0, BORDER_CONSTANT);
    filter2D(img, dxx, CV_64F, kernaldxx, Point(-1, -1), 0, BORDER_CONSTANT);
    filter2D(img, dyy, CV_64F, kernaldxx.t(), Point(-1, -1), 0, BORDER_CONSTANT);
    filter2D(img, dxy, CV_64F, kernaldxy, Point(-1, -1), 0, BORDER_CONSTANT);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img_threshold.at<uchar>(i, j) >= 220)
            {
                Mat hessian = Mat::zeros(2, 2, CV_64FC1);
                hessian.at<double>(0, 0) = dxx.at<double>(i, j);
                hessian.at<double>(0, 1) = dxy.at<double>(i, j);
                hessian.at<double>(1, 0) = dxy.at<double>(i, j);
                hessian.at<double>(1, 1) = dyy.at<double>(i, j);
                double Dxx = hessian.at<double>(0, 0);
                double Dxy = hessian.at<double>(1, 0);
                double Dyy = hessian.at<double>(1, 1);
                double tmp = sqrt(pow((Dxx - Dyy), 2) + 4 * pow(Dxy, 2));
                double v2x = 2 * Dxy;
                double v2y = Dyy - Dxx + tmp;
                double mag = sqrt(v2x * v2x + v2y * v2y);
                if (mag != 0)
                {
                    v2x = v2x / mag;
                    v2y = v2y / mag;
                }
                double v1x = -v2y;
                double v1y = v2x;
                double mu1 = 0.5 * (Dxx + Dyy + tmp);
                double mu2 = 0.5 * (Dxx + Dyy - tmp);
                double nx, ny;
                if (mu1 < mu2)
                {
                    nx = v2x;
                    ny = v2y;
                }
                else
                {
                    nx = v1x;
                    ny = v1y;
                }

                double t = -(nx * dx.at<double>(i, j) + ny * dy.at<double>(i, j)) / (nx * nx * dxx.at<double>(i, j) + 2 * nx * ny * dxy.at<double>(i, j) + ny * ny * dyy.at<double>(i, j));

                if (fabs(t * nx) <= 0.6 && fabs(t * ny) <= 0.6)
                {
                    double point_x = double(i) + 1.0 - t * nx;
                    double point_y = double(j) + 1.0 - t * ny;
                    Point2d line_point(point_y, point_x);
                    POINT.push_back(line_point);
                }
            }
        }
    }
    return POINT;
}
