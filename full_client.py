from __future__ import print_function

import os
import argparse
import grpc

import predict_search_pb2
import predict_search_pb2_grpc


def run(host, port):
    channel = grpc.insecure_channel('%s:%d' % (host, port))
    stub = predict_search_pb2_grpc.PredictorStub(channel)

    # # %% 预测请求：可展示中间输出结果
    # request = predict_search_pb2.PredictRequest(
    #     B_yepian_da = 14.005852,
    #     B_yepian_xiao = 24.795934,
    #     B_yepian_sui = 7.780687,
    #     H_yepian_da = 9.98,
    #     H_yepian_xiao = 31.23,
    #     H_yepian_sui = 6.76,
    #     tianchongzhi = 4.6,
    #     hanshuilv = 12.6,
    #     phanshuilv = 12.6, 
    #     redun_id='1100',
    # )
    # response = stub.predict(request)

    # print(f'预测结果：\n + \
    #     {response.B_qiesihou.chang, response.B_qiesihou.zhong, response.B_qiesihou.duan, response.B_qiesihou.sui}\n + \
    #     {response.H_qiesihou.chang, response.H_qiesihou.zhong, response.H_qiesihou.duan, response.H_qiesihou.sui}\n + \
    #     {response.B_fengxuanhou.chang, response.B_fengxuanhou.zhong, response.B_fengxuanhou.duan, response.B_fengxuanhou.sui}\n + \
    #     {response.H_lengquehou.chang, response.H_lengquehou.zhong, response.H_lengquehou.duan, response.H_lengquehou.sui}\n + \
    #     {response.chanpeihou.chang, response.chanpeihou.zhong, response.chanpeihou.duan, response.chanpeihou.sui}\n + \
    #     {response.xizu}'
    # )

    # %% 反推请求
    request = predict_search_pb2.SearchRequest(
        xizu = 1180,
        tianchongzhi = 4.6,
        hanshuilv = 12.6,
        phanshuilv = 12.6, 
        type_search = 'H',

        B_yepian_da = 12.3,
        B_yepian_xiao = 22.3,
        B_yepian_sui = 9.3,
        H_yepian_da = 19.3,
        H_yepian_xiao = 32.3,
        H_yepian_sui = 11.3,

        redun_id='1101',
    )
    response = stub.search(request)
    print(f'反推结果：\n + \
        {request.xizu}\n + \
        {response.B_module.da, 100 - (response.B_module.da+response.B_module.xiao+response.B_module.sui), response.B_module.xiao, response.B_module.sui}\n + \
        {response.H_module.da, 100 - (response.H_module.da+response.H_module.xiao+response.H_module.sui), response.H_module.xiao, response.H_module.sui}\n + \
        {response.close_xizu}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='host name', default='localhost', type=str)
    parser.add_argument('--port', help='port number', default=50052, type=int)
    args = parser.parse_args()
    run(args.host, args.port)
