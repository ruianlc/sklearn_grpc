import os
import sys
from concurrent import futures
import time
import argparse
import traceback
import multiprocessing

import grpc

import predict_search_pb2
import predict_search_pb2_grpc


def ip_predict(request):
    print(sys.path)
    import whole_process_model
    print(sys.modules['whole_process_model'])
    print('开始预测...')
    try:
        result = whole_process_model.main_predict(
            request.B_yepian_da, 
            request.B_yepian_xiao, 
            request.B_yepian_sui, 
            request.H_yepian_da, 
            request.H_yepian_xiao, 
            request.H_yepian_sui, 
            request.tianchongzhi, 
            request.hanshuilv, 
            request.phanshuilv,
        )
    except Exception as e:
        raise e
    finally:
        if 'whole_process_model' in sys.modules:
            print('删除环境变量：', sys.modules['whole_process_model'])
            sys.modules.pop('whole_process_model')
    return result

def ip_search(request):
    print(sys.path)
    import whole_process_model
    print(sys.modules['whole_process_model'])
    try:
        result = whole_process_model.main_search(
            request.xizu,
            request.tianchongzhi, 
            request.hanshuilv, 
            request.phanshuilv,
            request.type_search,
            request.B_yepian_da,
            request.B_yepian_xiao,
            request.B_yepian_sui,
            request.H_yepian_da,
            request.H_yepian_xiao,
            request.H_yepian_sui,
        )
    except Exception as e:
        raise e
    finally:
        if 'whole_process_model' in sys.modules:
            print('删除环境变量：', sys.modules['whole_process_model'])
            sys.modules.pop('whole_process_model')
    return result

class Predictor(predict_search_pb2_grpc.PredictorServicer):

    def predict(self, request, context):
        is_error = False
        error_message = ''
        try:
            print('当前工作路径为:', os.getcwd())

            pool = multiprocessing.Pool()
            result = pool.apply_async(ip_predict,args=(request,))
            pool.close()
            pool.join()
            result = result.get()

            print('预测完成')
        except Exception as e:
            print(e)
            error_message = traceback.format_exc()
            is_error = True
            result = ''
            print(error_message)

        response = predict_search_pb2.PredictResponse(
            redun_id=request.redun_id, 
            is_error=is_error, 
            error_message=error_message, 
            B_qiesihou=result[0], 
            H_qiesihou=result[1], 
            B_fengxuanhou=result[2], 
            H_lengquehou=result[3], 
            chanpeihou=result[4],
            xizu = result[5],
        )

        return response

    def search(self, request, context):
        is_error = False
        error_message = ''
        try:
            print('当前工作路径为:', os.getcwd())

            pool = multiprocessing.Pool()
            result = pool.apply_async(ip_search,args=(request,))
            pool.close()
            pool.join()
            result = result.get()

            print('预测完成')
        except Exception as e:
            print(e)
            error_message = traceback.format_exc()
            is_error = True
            result = ''
            print(error_message)

        response = predict_search_pb2.SearchResponse(
            redun_id=request.redun_id, 
            is_error=is_error, 
            error_message=error_message, 
            tianchongzhi=result[0],
            hanshuilv=result[1],
            phanshuilv=result[2],
            B_module=result[3],
            H_module=result[4],
            close_xizu=result[5],
        )

        return response

def serve(port, max_workers):
    while True:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        predict_search_pb2_grpc.add_PredictorServicer_to_server(Predictor(), server)
        server.add_insecure_port('[::]:{port}'.format(port=port))
        server.start()

        print(f"Server started, listening on {port}")
        server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', type=int, help='port number', required=False, default=50052)
    parser.add_argument('--max_workers', type=int, help='# max workers', required=False, default=10)
    args = parser.parse_args()

    serve(port=args.port, max_workers=args.max_workers)
