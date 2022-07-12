/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv_demo.h"

#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"

using namespace std;
using namespace cv;

static IVE_SRC_IMAGE_S pstSrc;
static IVE_DST_IMAGE_S pstDst;
static IVE_CSC_CTRL_S stCscCtrl;

typedef struct hiSAMPLE_IVE_IMAGE_INFO_S {
    IVE_IMAGE_S stSrc;
    IVE_IMAGE_S stDst;
    FILE *pFpSrc;
    FILE *pFpDst;
} SAMPLE_IVE_IMAGE_INFO_S;

static SAMPLE_IVE_IMAGE_INFO_S s_stimg;

static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
    VIDEO_FRAME_INFO_S *srcFrame)
{
    pstSrc->enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc->au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc->au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc->au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2]; // 2: Image data virtual address

    pstSrc->au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc->au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc->au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2]; // 2: Image data physical address

    pstSrc->au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc->au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc->au32Stride[2] = srcFrame->stVFrame.u32Stride[2]; // 2: Image data span

    pstSrc->u32Width = srcFrame->stVFrame.u32Width;
    pstSrc->u32Height = srcFrame->stVFrame.u32Height;

    pstDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst->u32Width = pstSrc->u32Width;
    pstDst->u32Height = pstSrc->u32Height;
    pstDst->au32Stride[0] = pstSrc->au32Stride[0];
    pstDst->au32Stride[1] = 0;
    pstDst->au32Stride[2] = 0; // 2: Image data span
}

static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB
    IveImageParamCfg(&pstSrc, &pstDst, srcFrame);

    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0],
        "User", HI_NULL, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        SAMPLE_PRT("HI_MPI_SYS_MmzFree err\n");
        return s32Ret;
    }

    s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0],
        pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }
    // 3: multiple
    memset_s((void *)pstDst.au64VirAddr[0], pstDst.u32Height*pstDst.au32Stride[0] * 3,
        0, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    HI_BOOL bInstant = HI_TRUE;

    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }

    if (HI_TRUE == bInstant) {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret) {
            usleep(100); // 100: usleep time
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;

    return HI_SUCCESS;
}

static HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame, Mat &dstMat)
{
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    int bufLen = w * h * 3;
    HI_U8 *srcRGB = NULL;
    IPC_IMAGE dstImage;
    if (yuvFrame2rgb(srcFrame, &dstImage) != HI_SUCCESS) {
        SAMPLE_PRT("yuvFrame2rgb err\n");
        return HI_FAILURE;
    }
    srcRGB = (HI_U8 *)dstImage.u64VirAddr;
    dstMat.create(h, w, CV_8UC3);
    memcpy_s(dstMat.data, bufLen * sizeof(HI_U8), srcRGB, bufLen * sizeof(HI_U8));
    HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
    return HI_SUCCESS;
}

HI_VOID jpg2yuv420(char* imgpath,char* yuvpath,unsigned int *Width, unsigned int *Height)              //yuv 420 sp (nv21) 切记！输入图像长宽必须是偶数
{   
    char* position;
    int cols,rows;
    SAMPLE_PRT("888888888888888888 yuvFrame2rgb err\n");
    memset_s(yuvpath, sizeof(*yuvpath), 0, sizeof(*yuvpath));
    strcpy(yuvpath,imgpath);
    position =strchr(yuvpath, '.');
    if(position)
    {
        memcpy(position + 1, "yuv",3);              //将路径的jpg改为yuv
    }
    else 
        return;
    // resize 图片到640*640
    Size dsize = Size(640, 384);
    cv::Mat Img = cv::imread(imgpath,3);
    cv::Mat Img_new;
	cv::resize(Img, Img_new, dsize, 0, 0, INTER_CUBIC);
    SAMPLE_PRT("999999999999999999999 yuvFrame2rgb err\n");

    FILE  *fp = fopen(yuvpath,"wb");
    if (Img_new.empty())
    {
        // std::cout << "empty!check your image";
        return;
    }
    cols = Img_new.cols;
    *Width   = Img_new.cols;
    rows = Img_new.rows;
    *Height  = Img_new.rows;

    int Yindex = 0;
    int UVindex = rows * cols;
    unsigned char* yuvbuff = (unsigned char *)malloc(rows*cols*1.5);
    cv::Mat NV21(rows+rows/2, cols, CV_8UC1);
    cv::Mat OpencvYUV;
    cv::Mat OpencvImg;
    cv::cvtColor(Img_new, OpencvYUV, COLOR_BGR2YUV_YV12);
    int UVRow{ 0 };
    for (int i=0;i<rows;i++)
    {
        for (int j=0;j<cols;j++)
        {
            uchar* YPointer = NV21.ptr<uchar>(i);

            int B = Img_new.at<cv::Vec3b>(i, j)[0];
            int G = Img_new.at<cv::Vec3b>(i, j)[1];
            int R = Img_new.at<cv::Vec3b>(i, j)[2];

            //计算Y的值
            int Y = (77 * R + 150 * G + 29 * B) >> 8;
            YPointer[j] = Y;
            yuvbuff[Yindex++] = (Y < 0) ? 0 : ((Y > 255) ? 255 : Y);
            uchar* UVPointer = NV21.ptr<uchar>(rows+i/2);
            //计算U、V的值，进行2x2的采样
            if (i%2==0&&(j)%2==0)
            {
                int U = ((-44 * R - 87 * G + 131 * B) >> 8) + 128;
                int V = ((131 * R - 110 * G - 21 * B) >> 8) + 128;
                UVPointer[j] = V;
                UVPointer[j+1] = U;
                yuvbuff[UVindex++] = (V < 0) ? 0 : ((V > 255) ? 255 : V);
                yuvbuff[UVindex++] = (U < 0) ? 0 : ((U > 255) ? 255 : U);
            }
        }
    }
    for (int i=0;i< 1.5 * rows * cols;i++)
    {
        fwrite(&yuvbuff[i], 1, 1, fp);
    }
    free(yuvbuff);
    fclose(fp);
    // cv::imshow("src", Img);//原图
    // cv::imshow("YUV", NV21);//转换后的图片
    // cv::imshow("opencv_YUV", OpencvYUV); //opencv转换后的图片
    // cv::imwrite("NV21.jpg", NV21);
    // cv::waitKey(30000);
    return;
}

// HI_S32 tennis_detect::TennisDetectLoad(uintptr_t* model)
// {
//     HI_S32 ret = 1;
//     *model = 1;
//     SAMPLE_PRT("TennisDetectLoad success\n");

//     return ret;
// }

// HI_S32 tennis_detect::TennisDetectUnload(uintptr_t model)
// {
//     model = 0;

//     return HI_SUCCESS;
// }

// HI_S32 tennis_detect::TennisDetectCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
// {
//     (void)model;
//     int ret = 0;
//     RectBox boxs[32] = {0}; // 32: TENNIS_OBJ_MAX
//     int j = 0;

//     Mat image;
//     frame2Mat(srcFrm, image);
//     if (image.size == 0) {
//         SAMPLE_PRT("image is null\n");
//         return HI_FAILURE;
//     }

//     Mat src = image;
//     Mat src1 = src.clone();
//     Mat dst, edge, gray, hsv;

//     dst.create(src1.size(), src1.type()); // Create a matrix of the same type and size as src (dst)

//     // The cvtColor operator is used to convert an image from one color space to another color space
//     cvtColor(src1, hsv, COLOR_BGR2HSV); // Convert original image to HSV image
    
//     // Binarize the hsv image, here is to binarize the green background,
//     // this parameter can be adjusted according to requirements
//     inRange(hsv, Scalar(31, 82, 68), Scalar(65, 248, 255), gray); // 31: B, 82: G, 68:R / 65: B, 248:G, 255:R

//     // Use canny operator for edge detection
//     // 3: threshold1, 9: threshold2, 3: apertureSize
//     Canny(gray, gray, 3, 9, 3);
//     vector<vector<Point>> contours;
//     findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
//     SAMPLE_PRT("contours.size():%d\n", contours.size());

//     for (int i = 0; i < (int)contours.size(); i++) {
//         if (contours.size() > 40) { // 40: contours.size() extremes
//             continue;
//         }

//         Rect ret1 = boundingRect(Mat(contours[i]));
//         ret1.x -= 5; // 5: x coordinate translation
//         ret1.y -= 5; // 5: y coordinate translation
//         ret1.width += 10; // 10: Rectangle width plus 10
//         ret1.height += 10; // 10: Rectangle height plus 10

//         // 20: Rectangle width and height pixel extremes
//         if ((ret1.width > 20) && (ret1.height > 20)) {
//             boxs[j].xmin = ret1.x * 3; // 3: optimized value
//             boxs[j].ymin = (int)(ret1.y * 2.25); // 2.25: optimized value
//             boxs[j].xmax = boxs[j].xmin + ret1.width * 3; // 3: optimized value
//             boxs[j].ymax = boxs[j].ymin + (int)ret1.height * 2.25; // 2.25: optimized value
//             j++;
//         }
//     }
//     // 25: detect boxesNum
//     if (j > 0 && j <= 25) {
//         SAMPLE_PRT("box num:%d\n", j);
//         MppFrmDrawRects(dstFrm, boxs, j, RGB888_RED, 2); // 2: DRAW_RETC_THICK
//     }

//     return ret;
// }

HI_S32 OpencvDemo(VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_PRT("i am 5");
    Mat image;
    frame2Mat(srcFrm, image);
    if (image.size == 0) {
        SAMPLE_PRT("image is null\n");
        return HI_FAILURE;
    }
    Mat src = image;
    Mat src1 = src.clone();
    Mat dst, edge, gray, hsv;

    dst.create(src1.size(), src1.type()); // Create a matrix of the same type and size as src (dst)

    // The cvtColor operator is used to convert an image from one color space to another color space
    // cvtColor(src1, hsv, COLOR_BGR2GRAY); // Convert original image to HSV image
    
    // Binarize the hsv image, here is to binarize the green background,
    // this parameter can be adjusted according to requirements
    // inRange(hsv, Scalar(31, 82, 68), Scalar(65, 248, 255), gray); // 31: B, 82: G, 68:R / 65: B, 248:G, 255:R

    // Use canny operator for edge detection
    // 3: threshold1, 9: threshold2, 3: apertureSize
    // Canny(hsv, gray, 3, 9, 3);

    cv::imwrite("./img/src.jpg", src1);
    SAMPLE_PRT("save one image");
    // unsigned int *width;
    // unsigned int *height;
    // *width = 640;
    // *height = 384;
    // jpg2yuv420("./img/src.jpg","./img/gray2.yuv",width, height);

    // HI_CHAR *pchSrcFileName = "./img/gray2.yuv";
    // HI_CHAR *pchDstFileName = "./img/gray_out.yuv";
    // HI_U32 u32Width = 640;
    // HI_U32 u32Height = 384;
    // (HI_VOID)memset_s(&s_stimg, sizeof(s_stimg), 0, sizeof(s_stimg));
    //         HI_S32 s32Ret = HI_SUCCESS;
    // (HI_VOID)memset_s(&s_stimg, sizeof(SAMPLE_IVE_IMAGE_INFO_S), 0, sizeof(SAMPLE_IVE_IMAGE_INFO_S));
    // s32Ret = SAMPLE_COMM_IVE_CreateImage(&(s_stimg.stSrc), IVE_IMAGE_TYPE_U8C1, u32Width, u32Height);
    // s32Ret = SAMPLE_COMM_IVE_CreateImage(&(s_stimg.stDst), IVE_IMAGE_TYPE_U8C1, u32Width, u32Height);
    // s_stimg.pFpSrc = fopen(pchSrcFileName, "rb");
    // s_stimg.pFpDst = fopen(pchDstFileName, "wb");
    // s32Ret = SAMPLE_COMM_IVE_ReadFile(&(s_stimg.stSrc), s_stimg.pFpSrc);
    // int ret;
    // // static VIDEO_FRAME_INFO_S frm;
    // ret = OrigImgToFrm(&s_stimg.stSrc, dstFrm);    
    return 0;
}