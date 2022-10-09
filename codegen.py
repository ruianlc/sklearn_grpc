"""
Generates protocol messages and gRPC stubs.
生成grpc代码，以proto文件为输入，输出两个py文件：
1. _pb2.py: contains our generated request and response classes 
2. _pb2_grpc.py: contains our generated client and server classes.
"""
from grpc.tools import protoc


protoc.main(
    (
        '',
        '-I=./protos',          # 指定proto所在目录
        '--python_out=.',       # 生成服务py文件所在目录
        '--grpc_python_out=.',  # 生成服务grpc py文件所在目录
        './protos/predict_search.proto', # 指定proto文件
    )
)