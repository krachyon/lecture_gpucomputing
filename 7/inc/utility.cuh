#pragma once

#include <vector_types.h>

__device__ inline bool operator==(float3 const& lhs, float3 const& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

__device__ inline float3 operator+(float3 const& lhs, float3 const& rhs)
{
return float3{lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z};
}

__device__ inline float3 operator-(float3 const& lhs, float3 const& rhs)
{
return float3{lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z};
}

__device__ inline float3 operator*(float lhs, float3 const& rhs)
{
    return float3{lhs*rhs.x, lhs*rhs.y, lhs*rhs.z};
}

__device__ inline float3 operator*(float3 const& lhs, float rhs)
{
    return float3{lhs.x*rhs, lhs.y*rhs, lhs.z*rhs};
}

__device__ inline float3 operator/(float lhs, float3 const& rhs)
{
    return (1.f/lhs)*rhs;
}

__device__ inline float3 operator/(float3 const& lhs, float rhs)
{
    return lhs*(1.f/rhs);
}

__device__ inline float3& operator+=(float3& lhs , float3 const& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

__device__ inline float norm_pow3(float3 const& vec)
{
    return vec.x*vec.x*vec.x + vec.y*vec.y*vec.y + vec.z*vec.z*vec.z;
}

inline __device__ __host__ uint32_t ceildiv(uint32_t x, uint32_t y) {
    // division instruction gives you a free modulo. So add one if not cleanly divisible. not that should matter...
    return x / y + (x % y != 0);
}

