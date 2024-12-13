#include "utils/utils.h"

#include <string>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

namespace inferx
{

void read_data_from_txt(const std::string& file_path, float* data, size_t size)
{
    std::ifstream file(file_path.c_str());

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        LOG(ERROR) << "Can't open file " << file_path;
    }

    std::string line;
    std::stringstream ss;

    // 读取整个文件内容
    ss << file.rdbuf();             // 将文件内容读入到字符串流中
    std::string numbers = ss.str(); // 获取字符串流内容

    std::istringstream iss(numbers);
    float value;
    int idx = 0;
    // 读取每个 float 值，按空格分隔
    while (iss >> value)
    {
        data[idx++] = value;
        // std::cout << value << std::endl;
    }

    file.close(); // 关闭文件
}
} // namespace inferx