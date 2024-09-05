#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>
#include <cstddef>

namespace inferx
{
namespace core
{
enum DataType : uint8_t
{
    DataTypeUnknown = 0,
    DataTypeFloat32 = 1, // 4
    DataTypeFloat64 = 2, // 8
    DataTypeInt8 = 3,    //
    DataTypeInt16 = 4,
    DataTypeInt32 = 5,
};

enum StatusCode : uint8_t
{
    Success = 0,
    Failed = 1,
    OutOfMemory = 2,
    InvalidValue = 3,
    NotImplemented = 4,
};

} // namespace core
} // namespace inferx

#endif