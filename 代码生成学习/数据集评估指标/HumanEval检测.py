import ast
import textwrap
import signal

# linux 下可以使用如下代码：


# 定义一个信号处理器，用于在超时时抛出异常
def timeout_handler(signum, frame):
    raise TimeoutError("Test execution exceeded time limit")


# 从给定代码字符串中提取指定函数的函数体
def extract_function_body(code, entry_point):
    try:
        # 将代码字符串解析为抽象语法树（AST）
        tree = ast.parse(code)

        # 遍历AST中的所有节点
        for node in ast.walk(tree):
            # 找到与 entry_point 名称匹配的函数定义节点
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                # 将函数体重新转换为代码字符串
                code = ast.unparse(node.body)
                # 设定缩进字符串（四个空格）
                indent_str = "    "
                # 为提取出的代码段增加缩进
                indented_code = textwrap.indent(text=code, prefix=indent_str)
                return indented_code
    except:
        # 如果解析失败，返回原始代码
        return code


# 检查并执行代码的函数
def check_code(prompt, final, test, entry_point):
    # 设置信号处理器，当信号为 SIGALRM 时调用 timeout_handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # 从 final 中提取 entry_point 对应的函数体
    final = extract_function_body(final, entry_point)

    # 如果成功提取到函数体，组合成完整代码
    if final != None:
        final_code = prompt + final
    else:
        final_code = prompt  # 如果未提取到函数体，使用 prompt 原始代码

    try:
        # 执行组合后的代码（prompt 和函数体）
        exec(final_code)
        print(final_code)  # 打印执行的代码（可用于调试）
    except:
        # 如果代码执行失败，打印错误信息并返回 False
        print("wrong code")
        return False

    # 设置10秒的超时时间，确保测试代码不会运行过长时间
    signal.alarm(10)

    # 执行测试代码，通常用于验证 prompt+final_code 是否符合预期
    exec(test)

    try:
        # 调用 `check` 函数来验证 entry_point 对应的函数是否正确
        locals()["check"]((locals()[entry_point]))
        print("Success")  # 如果成功通过验证，打印成功信息
        return True  # 返回 True 表示代码正确
    except Exception as e:
        # 如果验证失败，捕获异常并返回 False
        # 可选地打印异常信息 (当前注释掉了)
        # print(e)
        return False
    finally:
        # 最终取消定时警报，无论是否抛出异常
        signal.alarm(0)  # 取消设置的定时器
