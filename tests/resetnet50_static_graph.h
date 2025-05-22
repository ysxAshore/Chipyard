#include <stdio.h>
#include "mt-queuelock.h"
// CONV_PARAMS(2) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 64, 	kh kw ic oc = 1 1 64 64, 	stride = 1, };	1(3136,64,64)1,padding=0,3136%64==0 	Input_size:0.19MB,weight_size:0.00MB,output_size:0.77MB,Mops:12.85MMAC
// CONV_PARAMS(3) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 64, 	kh kw ic oc = 3 3 64 64, 	stride = 1, };	3(3136,64,64)1,padding=1,3136%64==0 	Input_size:0.19MB,weight_size:0.04MB,output_size:0.77MB,Mops:115.61MMAC
// CONV_PARAMS(4) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 256, 	kh kw ic oc = 1 1 64 256, 	stride = 1, };	1(3136,256,64)1,padding=0,3136%64==0 	Input_size:0.19MB,weight_size:0.02MB,output_size:3.06MB,Mops:51.38MMAC
// CONV_PARAMS(5) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 256, 	kh kw ic oc = 1 1 64 256, 	stride = 1, };	1(3136,256,64)1,padding=0,3136%64==0 	Input_size:0.19MB,weight_size:0.02MB,output_size:3.06MB,Mops:51.38MMAC
// CONV_PARAMS(6) = {	ih iw ic = 56 56 256, 	oh ow oc = 56 56 64, 	kh kw ic oc = 1 1 256 64, 	stride = 1, };	1(3136,64,256)1,padding=0,3136%64==0 	Input_size:0.77MB,weight_size:0.02MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(7) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 64, 	kh kw ic oc = 3 3 64 64, 	stride = 1, };	3(3136,64,64)1,padding=1,3136%64==0 	Input_size:0.19MB,weight_size:0.04MB,output_size:0.77MB,Mops:115.61MMAC
// CONV_PARAMS(8) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 256, 	kh kw ic oc = 1 1 64 256, 	stride = 1, };	1(3136,256,64)1,padding=0,3136%64==0 	Input_size:0.19MB,weight_size:0.02MB,output_size:3.06MB,Mops:51.38MMAC
// CONV_PARAMS(9) = {	ih iw ic = 56 56 256, 	oh ow oc = 56 56 64, 	kh kw ic oc = 1 1 256 64, 	stride = 1, };	1(3136,64,256)1,padding=0,3136%64==0 	Input_size:0.77MB,weight_size:0.02MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(10) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 64, 	kh kw ic oc = 3 3 64 64, 	stride = 1, };	3(3136,64,64)1,padding=1,3136%64==0 	Input_size:0.19MB,weight_size:0.04MB,output_size:0.77MB,Mops:115.61MMAC
// CONV_PARAMS(11) = {	ih iw ic = 56 56 64, 	oh ow oc = 56 56 256, 	kh kw ic oc = 1 1 64 256, 	stride = 1, };	1(3136,256,64)1,padding=0,3136%64==0 	Input_size:0.19MB,weight_size:0.02MB,output_size:3.06MB,Mops:51.38MMAC
// CONV_PARAMS(12) = {	ih iw ic = 56 56 256, 	oh ow oc = 56 56 128, 	kh kw ic oc = 1 1 256 128, 	stride = 1, };	1(3136,128,256)1,padding=0,3136%64==0 	Input_size:0.77MB,weight_size:0.03MB,output_size:1.53MB,Mops:102.76MMAC
// CONV_PARAMS(13) = {	ih iw ic = 56 56 128, 	oh ow oc = 28 28 128, 	kh kw ic oc = 3 3 128 128, 	stride = 2, };	3(784,128,128)2,padding=1,784%64==16 	Input_size:0.38MB,weight_size:0.14MB,output_size:0.38MB,Mops:115.61MMAC
// CONV_PARAMS(14) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 512, 	kh kw ic oc = 1 1 128 512, 	stride = 1, };	1(784,512,128)1,padding=0,784%64==16 	Input_size:0.10MB,weight_size:0.06MB,output_size:1.53MB,Mops:51.38MMAC
// CONV_PARAMS(15) = {	ih iw ic = 56 56 256, 	oh ow oc = 28 28 512, 	kh kw ic oc = 1 1 256 512, 	stride = 2, };	1(784,512,256)2,padding=0,784%64==16 	Input_size:0.77MB,weight_size:0.12MB,output_size:1.53MB,Mops:102.76MMAC
// CONV_PARAMS(16) = {	ih iw ic = 28 28 512, 	oh ow oc = 28 28 128, 	kh kw ic oc = 1 1 512 128, 	stride = 1, };	1(784,128,512)1,padding=0,784%64==16 	Input_size:0.38MB,weight_size:0.06MB,output_size:0.38MB,Mops:51.38MMAC
// CONV_PARAMS(17) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 128, 	kh kw ic oc = 3 3 128 128, 	stride = 1, };	3(784,128,128)1,padding=1,784%64==16 	Input_size:0.10MB,weight_size:0.14MB,output_size:0.38MB,Mops:115.61MMAC
// CONV_PARAMS(18) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 512, 	kh kw ic oc = 1 1 128 512, 	stride = 1, };	1(784,512,128)1,padding=0,784%64==16 	Input_size:0.10MB,weight_size:0.06MB,output_size:1.53MB,Mops:51.38MMAC
// CONV_PARAMS(19) = {	ih iw ic = 28 28 512, 	oh ow oc = 28 28 128, 	kh kw ic oc = 1 1 512 128, 	stride = 1, };	1(784,128,512)1,padding=0,784%64==16 	Input_size:0.38MB,weight_size:0.06MB,output_size:0.38MB,Mops:51.38MMAC
// CONV_PARAMS(20) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 128, 	kh kw ic oc = 3 3 128 128, 	stride = 1, };	3(784,128,128)1,padding=1,784%64==16 	Input_size:0.10MB,weight_size:0.14MB,output_size:0.38MB,Mops:115.61MMAC
// CONV_PARAMS(21) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 512, 	kh kw ic oc = 1 1 128 512, 	stride = 1, };	1(784,512,128)1,padding=0,784%64==16 	Input_size:0.10MB,weight_size:0.06MB,output_size:1.53MB,Mops:51.38MMAC
// CONV_PARAMS(22) = {	ih iw ic = 28 28 512, 	oh ow oc = 28 28 128, 	kh kw ic oc = 1 1 512 128, 	stride = 1, };	1(784,128,512)1,padding=0,784%64==16 	Input_size:0.38MB,weight_size:0.06MB,output_size:0.38MB,Mops:51.38MMAC
// CONV_PARAMS(23) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 128, 	kh kw ic oc = 3 3 128 128, 	stride = 1, };	3(784,128,128)1,padding=1,784%64==16 	Input_size:0.10MB,weight_size:0.14MB,output_size:0.38MB,Mops:115.61MMAC
// CONV_PARAMS(24) = {	ih iw ic = 28 28 128, 	oh ow oc = 28 28 512, 	kh kw ic oc = 1 1 128 512, 	stride = 1, };	1(784,512,128)1,padding=0,784%64==16 	Input_size:0.10MB,weight_size:0.06MB,output_size:1.53MB,Mops:51.38MMAC
// CONV_PARAMS(25) = {	ih iw ic = 28 28 512, 	oh ow oc = 28 28 256, 	kh kw ic oc = 1 1 512 256, 	stride = 1, };	1(784,256,512)1,padding=0,784%64==16 	Input_size:0.38MB,weight_size:0.12MB,output_size:0.77MB,Mops:102.76MMAC
// CONV_PARAMS(26) = {	ih iw ic = 28 28 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 2, };	3(196,256,256)2,padding=1,196%64==4 	Input_size:0.19MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(27) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(28) = {	ih iw ic = 28 28 512, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 512 1024, 	stride = 2, };	1(196,1024,512)2,padding=0,196%64==4 	Input_size:0.38MB,weight_size:0.50MB,output_size:0.77MB,Mops:102.76MMAC
// CONV_PARAMS(29) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 256, 	kh kw ic oc = 1 1 1024 256, 	stride = 1, };	1(196,256,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.25MB,output_size:0.19MB,Mops:51.38MMAC
// CONV_PARAMS(30) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 1, };	3(196,256,256)1,padding=1,196%64==4 	Input_size:0.05MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(31) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(32) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 256, 	kh kw ic oc = 1 1 1024 256, 	stride = 1, };	1(196,256,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.25MB,output_size:0.19MB,Mops:51.38MMAC
// CONV_PARAMS(33) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 1, };	3(196,256,256)1,padding=1,196%64==4 	Input_size:0.05MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(34) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(35) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 256, 	kh kw ic oc = 1 1 1024 256, 	stride = 1, };	1(196,256,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.25MB,output_size:0.19MB,Mops:51.38MMAC
// CONV_PARAMS(36) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 1, };	3(196,256,256)1,padding=1,196%64==4 	Input_size:0.05MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(37) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(38) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 256, 	kh kw ic oc = 1 1 1024 256, 	stride = 1, };	1(196,256,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.25MB,output_size:0.19MB,Mops:51.38MMAC
// CONV_PARAMS(39) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 1, };	3(196,256,256)1,padding=1,196%64==4 	Input_size:0.05MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(40) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(41) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 256, 	kh kw ic oc = 1 1 1024 256, 	stride = 1, };	1(196,256,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.25MB,output_size:0.19MB,Mops:51.38MMAC
// CONV_PARAMS(42) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 256, 	kh kw ic oc = 3 3 256 256, 	stride = 1, };	3(196,256,256)1,padding=1,196%64==4 	Input_size:0.05MB,weight_size:0.56MB,output_size:0.19MB,Mops:115.61MMAC
// CONV_PARAMS(43) = {	ih iw ic = 14 14 256, 	oh ow oc = 14 14 1024, 	kh kw ic oc = 1 1 256 1024, 	stride = 1, };	1(196,1024,256)1,padding=0,196%64==4 	Input_size:0.05MB,weight_size:0.25MB,output_size:0.77MB,Mops:51.38MMAC
// CONV_PARAMS(44) = {	ih iw ic = 14 14 1024, 	oh ow oc = 14 14 512, 	kh kw ic oc = 1 1 1024 512, 	stride = 1, };	1(196,512,1024)1,padding=0,196%64==4 	Input_size:0.19MB,weight_size:0.50MB,output_size:0.38MB,Mops:102.76MMAC
// CONV_PARAMS(45) = {	ih iw ic = 14 14 512, 	oh ow oc = 7 7 512, 	kh kw ic oc = 3 3 512 512, 	stride = 2, };	3(49,512,512)2,padding=1,49%64==49 	Input_size:0.10MB,weight_size:2.25MB,output_size:0.10MB,Mops:115.61MMAC
// CONV_PARAMS(46) = {	ih iw ic = 7 7 512, 	oh ow oc = 7 7 2048, 	kh kw ic oc = 1 1 512 2048, 	stride = 1, };	1(49,2048,512)1,padding=0,49%64==49 	Input_size:0.02MB,weight_size:1.00MB,output_size:0.38MB,Mops:51.38MMAC
// CONV_PARAMS(47) = {	ih iw ic = 14 14 1024, 	oh ow oc = 7 7 2048, 	kh kw ic oc = 1 1 1024 2048, 	stride = 2, };	1(49,2048,1024)2,padding=0,49%64==49 	Input_size:0.19MB,weight_size:2.00MB,output_size:0.38MB,Mops:102.76MMAC
// CONV_PARAMS(48) = {	ih iw ic = 7 7 2048, 	oh ow oc = 7 7 512, 	kh kw ic oc = 1 1 2048 512, 	stride = 1, };	1(49,512,2048)1,padding=0,49%64==49 	Input_size:0.10MB,weight_size:1.00MB,output_size:0.10MB,Mops:51.38MMAC
// CONV_PARAMS(49) = {	ih iw ic = 7 7 512, 	oh ow oc = 7 7 512, 	kh kw ic oc = 3 3 512 512, 	stride = 1, };	3(49,512,512)1,padding=1,49%64==49 	Input_size:0.02MB,weight_size:2.25MB,output_size:0.10MB,Mops:115.61MMAC
// CONV_PARAMS(50) = {	ih iw ic = 7 7 512, 	oh ow oc = 7 7 2048, 	kh kw ic oc = 1 1 512 2048, 	stride = 1, };	1(49,2048,512)1,padding=0,49%64==49 	Input_size:0.02MB,weight_size:1.00MB,output_size:0.38MB,Mops:51.38MMAC
// CONV_PARAMS(51) = {	ih iw ic = 7 7 2048, 	oh ow oc = 7 7 512, 	kh kw ic oc = 1 1 2048 512, 	stride = 1, };	1(49,512,2048)1,padding=0,49%64==49 	Input_size:0.10MB,weight_size:1.00MB,output_size:0.10MB,Mops:51.38MMAC
// CONV_PARAMS(52) = {	ih iw ic = 7 7 512, 	oh ow oc = 7 7 512, 	kh kw ic oc = 3 3 512 512, 	stride = 1, };	3(49,512,512)1,padding=1,49%64==49 	Input_size:0.02MB,weight_size:2.25MB,output_size:0.10MB,Mops:115.61MMAC
// CONV_PARAMS(53) = {	ih iw ic = 7 7 512, 	oh ow oc = 7 7 2048, 	kh kw ic oc = 1 1 512 2048, 	stride = 1, };	1(49,2048,512)1,padding=0,49%64==49 	Input_size:0.02MB,weight_size:1.00MB,output_size:0.38MB,Mops:51.38MMAC


typedef struct CONV_PARAMS {
    int layer_id;
    int ih, iw, ic;
    int oh, ow, oc;
    int kh, kw;
    int stride;
}conv_params;

//树的结构，方便层级遍历
typedef struct GRAPH_NODE {
    conv_params conv_param;
    struct GRAPH_NODE *next_node;
}graph_node;

typedef struct GRAPH {
    graph_node *head;
    graph_node *tail;
}resnet50_graph;

int ih_list[54] = {
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 7 ,7 ,7 ,7 ,
    7 ,7 ,7 ,7 ,7 ,7 ,
};
int iw_list[54] = {
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 7 ,7 ,7 ,7 ,
    7 ,7 ,7 ,7 ,7 ,7 ,
};
int ic_list[54] = {
    64, 64, 64, 64, 64, 256, 256, 256, 256, 256,
    128, 128, 128, 128, 512, 512, 512, 512, 512, 512,
    128, 128, 128, 128, 512, 512, 512, 512, 256, 256,
    256, 256, 1024,1024,1024,1024,1024 ,1024 ,512 ,512 ,
    2048 ,2048 ,2048 ,2048 ,2048 ,2048 ,
};
int oc_list[54] = {
    64, 64, 64, 256, 256, 64, 64, 256, 256, 64,
    128, 128, 128, 512, 512, 512, 512, 512, 512, 128,
    128, 128, 512, 512, 512, 512, 256, 256, 256, 256,
    1024 ,1024 ,1024 ,1024 ,1024 ,1024 ,256 ,256 ,
    512 ,512 ,2048 ,2048 ,2048 ,2048 ,
};
int kh_list[54] = {
    1, 3, 1, 1, 3, 1, 3, 1, 3, 3,
    3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 1 ,1 ,
    1 ,1 ,1 ,1 ,3 ,3 ,3 ,3 ,
    3 ,3 ,3 ,3 ,1 ,1 ,
};
int kw_list[54] = {
    1, 3, 1, 1, 3, 1, 3, 1, 3, 3,
    3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 1 ,1 ,
    1 ,1 ,1 ,1 ,3 ,3 ,3 ,3 ,
    3 ,3 ,3 ,3 ,1 ,1 ,
};
int stride_list[54] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,
    2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,
    1 ,1 ,1 ,1 ,1 ,1 ,
};
queue_lock_t console_lock;

resnet50_graph *init_resnet50_graph() {
    queue_lock_init(&console_lock);
    resnet50_graph *graph = (resnet50_graph *)malloc(sizeof(resnet50_graph));
    graph->head = NULL;
    graph->tail = NULL;

    for (int i = 0; i < 54; i++) {
        graph_node *node = (graph_node *)malloc(sizeof(graph_node));
        node->conv_param.layer_id = i;
        node->conv_param.ih = ih_list[i];
        node->conv_param.iw = iw_list[i];
        node->conv_param.ic = ic_list[i];
        node->conv_param.oh = ih_list[i];
        node->conv_param.ow = iw_list[i];
        node->conv_param.oc = oc_list[i];
        node->conv_param.kh = kh_list[i];
        node->conv_param.kw = kw_list[i];
        node->conv_param.stride = stride_list[i];
        node->next_node = NULL;

        if (graph->head == NULL) {
            graph->head = node;
            graph->tail = node;
        } else {
            graph->tail->next_node = node;
            graph->tail = node;
        }
    }
    return graph;
}

//next task -> get next node

graph_node *get_next_node(resnet50_graph *graph, graph_node *current_node) {
    if (current_node == NULL) {
        return NULL;
    }
    return current_node->next_node;
}



void resenet50_caculate_func(int thread_id, void *thread_params) {
    // thread_local conv_params;
    conv_params *params = (conv_params *)thread_params;

    //获取hartid
    int hart_id = read_csr(mhartid);
    // 获得锁然后输出参数
    queue_lock_acquire(&console_lock);
    printf("<core %d>Thread %d:do layer[%d] ih = %d, iw = %d, ic = %d, oh = %d, ow = %d, oc = %d, kh = %d, kw = %d, stride = %d\n",
              hart_id,
           thread_id,
           params->layer_id,
           params->ih,
           params->iw,
           params->ic,
           params->oh,
           params->ow,
           params->oc,
           params->kh,
           params->kw,
           params->stride);
    // 释放锁
    queue_lock_release(&console_lock);
}


//根据当前节点获取下一个节点的任务(一个函数指针)，如果没有下一个节点，则返回NULL
void* get_current_task(resnet50_graph *graph, graph_node *current_node) {
    // current_node = get_next_node(graph, current_node);
    if (current_node == graph->tail) {
        return NULL;
    }
    else {
        return (void *)resenet50_caculate_func;
    }
}