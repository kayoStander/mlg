#pragma once

#include <cctype>
#include <cmath>
#include <cstring>

namespace {
template <typename T> class Vector {
public:
  virtual void sqrt() {};
  virtual void operator+() {};
  virtual void operator-() {};
  virtual T operator[](int index) { return T(0.0); };
  virtual const T operator[](int index) const { return T(0.0); };

private:
protected:
  virtual ~Vector(){};
};
class Matrix {
public:
private:
protected:
  virtual ~Matrix(){};
};
} // namespace

namespace Math {

#ifndef NOINTTEMPLATE
using int8 = int8_t;
using c_int8 = const int8_t;
using int32 = int32_t;
using c_int32 = const int32_t;
#endif

namespace Constant {
template <typename T> constexpr T TWOPI() {
  return static_cast<T>(6.28318530718);
}
template <typename T> constexpr T PI() { return static_cast<T>(3.1415926535); }
template <typename T> constexpr T HALFPI() {
  return static_cast<T>(1.5707963267);
}
template <typename T> constexpr T ONEOVERPI() {
  return static_cast<T>(0.3183098861);
}

template <typename T> constexpr T EULER() {
  return static_cast<T>(0.5772156649);
}
template <typename T> constexpr T GOLDENRATIO() {
  return static_cast<T>(1.618);
}
} // namespace Constant

template <typename T> class Vec2 : public Vector<T> {
public:
  Vec2() : x(static_cast<T>(0.0)), y(static_cast<T>(0.0)) {}
  Vec2(const T x, const T y) : x(x), y(y) {}
  Vec2(const Vec2<T> &xy) : x(xy.x), y(xy.y) {}

  void sqrt() {
    x = static_cast<T>(std::sqrt(x));
    y = static_cast<T>(std::sqrt(y));
  }
  void operator+(T value) {
    x += value;
    y += value;
  }
  void operator-(T value) {
    x -= value;
    y -= value;
  }

  T operator[](int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    }
    return static_cast<T>(0.0);
  }

  const T operator[](int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    }
    return static_cast<T>(0.0);
  }

  union {
    struct {
      T x, y;
    };
    struct {
      Vec2<T> &vec2XYZ;
    } vec2XYZ;
  };

private:
};

template <typename T> class Vec3 : public Vector<T> {
public:
  Vec3()
      : x(static_cast<T>(0.0)), y(static_cast<T>(0.0)), z(static_cast<T>(0.0)) {
  }
  Vec3(const T x, const T y, const T z) : x(x), y(y), z(z) {}
  Vec3(const Vec3<T> &xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}
  Vec3(const Vec2<T> &xy, const T z) : x(xy.x), y(xy.y), z(z) {}
  Vec3(const T x, const Vec2<T> &yz) : x(x), y(yz.x), z(yz.y) {}

  void sqrt() {
    x = static_cast<T>(std::sqrt(x));
    y = static_cast<T>(std::sqrt(y));
    z = static_cast<T>(std::sqrt(z));
  }
  void operator+(T value) {
    x += value;
    y += value;
    z += value;
  }
  void operator-(T value) {
    x -= value;
    y -= value;
    z -= value;
  }

  T operator[](int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    return static_cast<T>(0.0);
  }

  const T operator[](int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    return static_cast<T>(0.0);
  }

  union {
    struct {
      T x, y, z;
    };
    struct {
      Vec3<T> &vecXYZ;
    };
    struct {
      Vec2<T> &vecXY;
      T z;
    } Vec2Z;
  };

private:
};

template <typename T> class Vec4 : public Vector<T> {
public:
  Vec4()
      : x(static_cast<T>(0.0)), y(static_cast<T>(0.0)), z(static_cast<T>(0.0)),
        w(static_cast<T>(0.0)) {}
  Vec4(const T x, const T y, const T z, const T w) : x(x), y(y), z(z), w(w) {}
  Vec4(const Vec4<T> &xyzw) : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}
  Vec4(const Vec2<T> &xy, const Vec2<T> &zw)
      : x(xy.x), y(xy.y), z(zw.x), w(zw.y) {}
  Vec4(const T x, const T y, const Vec2<T> &zw)
      : x(x), y(y), z(zw.x), w(zw.y) {}
  Vec4(const Vec2<T> &xy, const T z, const T w)
      : x(xy.x), y(xy.y), z(z), w(w) {}
  Vec4(const Vec3<T> &xyz, const T w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}
  Vec4(const T x, const Vec3<T> &yzw) : x(x), y(yzw.y), z(yzw.z), w(yzw.w) {}

  void sqrt() {
    x = static_cast<T>(std::sqrt(x));
    y = static_cast<T>(std::sqrt(y));
    z = static_cast<T>(std::sqrt(z));
    w = static_cast<T>(std::sqrt(w));
  }
  void operator+(T value) {
    x += value;
    y += value;
    z += value;
    w += value;
  }
  void operator-(T value) {
    x -= value;
    y -= value;
    z -= value;
    w -= value;
  }

  T operator[](int index) {
    // return index < 4 ? *(T *)((char *)this + sizeof(T)) :
    // static_cast<T>(0.0f);
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    case 3:
      return w;
    }
    return static_cast<T>(0.0);
  }

  const T operator[](int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    case 3:
      return w;
    }
    return static_cast<T>(0.0);
  }

  union {
    struct {
      T x, y, z, w;
    };
    struct {
      Vec4<T> &vecXYZW;
    };
    struct {
      Vec2<T> &vecXY;
      Vec2<T> &vecZW;
    };
    struct {
      Vec3<T> &vecXYZ;
      T w;
    } vec3W;
  };

private:
};

template <typename T> class Mat4x4 : Matrix {

  enum MAT4CONSTRUCTTYPE { DIAGONAL = 0, BLOCK = 1, VEC4 = 2 };

public:
  Mat4x4() {
    constructorBLOCK(static_cast<T>(0.0), static_cast<T>(0.0),
                     static_cast<T>(0.0), static_cast<T>(0.0));
  }
  Mat4x4(T x, T y, T z, T w,
         MAT4CONSTRUCTTYPE ConstructType = MAT4CONSTRUCTTYPE::DIAGONAL) {
    switch (ConstructType) {
    case MAT4CONSTRUCTTYPE::DIAGONAL:
      ConstructorDIAGONAL(x, y, z, w);
      break;
    case MAT4CONSTRUCTTYPE::BLOCK:
      ConstructorBLOCK(x, y, z, w);
      break;
    }
  }
  Mat4x4(Vec4<T> &xyzw,
         MAT4CONSTRUCTTYPE ConstructType = MAT4CONSTRUCTTYPE::DIAGONAL) {
    switch (ConstructType) {
    case MAT4CONSTRUCTTYPE::DIAGONAL:
      ConstructorDIAGONAL(xyzw.x, xyzw.y, xyzw.z, xyzw.w);
      break;
    case MAT4CONSTRUCTTYPE::BLOCK:
      ConstructorBLOCK(xyzw.x, xyzw.y, xyzw.z, xyzw.w);
      break;
    case MAT4CONSTRUCTTYPE::VEC4:
      ConstructorVEC4(xyzw.x, xyzw.y, xyzw.z, xyzw.w);
      break;
    }
  }
  Mat4x4(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w,
         MAT4CONSTRUCTTYPE ConstructType = MAT4CONSTRUCTTYPE::DIAGONAL) {
    switch (ConstructType) {
    case MAT4CONSTRUCTTYPE::DIAGONAL:
      ConstructorDIAGONAL(x, y, z, w);
      break;
    case MAT4CONSTRUCTTYPE::BLOCK:
      ConstructorBLOCK(x, y, z, w);
      break;
    case MAT4CONSTRUCTTYPE::VEC4:
      ConstructorVEC4(x, y, z, w);
      break;
    }
  }

  T *operator[](int index) { return mat4[index]; }

  union {
    struct {
      T mat4[4][4];
    };
    struct {
      Vec4<T> x;
      Vec4<T> y;
      Vec4<T> z;
      Vec4<T> w;
    };
  };

private:
  void ConstructorVEC4(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (int8 i = 0; i < 4; i++) {
      mat4[0][i] = x[i];
      mat4[1][i] = y[i];
      mat4[2][i] = z[i];
      mat4[3][i] = w[i];
    }
  }
  void ConstructorVEC4(Vec4<T> xyzw) {
    for (int8 i = 0; i < 4; i++) {
      mat4[0][i] = xyzw.x;
      mat4[1][i] = xyzw.y;
      mat4[2][i] = xyzw.z;
      mat4[3][i] = xyzw.w;
    }
  }
  void ConstructorVEC4(T x, T y, T z, T w) {
    for (int8 i = 0; i < 4; i++) {
      mat4[0][i] = x;
      mat4[1][i] = y;
      mat4[2][i] = z;
      mat4[3][i] = w;
    }
  }

  void ConstructorBLOCK(Vec4<T> xyzw) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0)   ? xyzw.x
                     : (j == 1) ? xyzw.y
                     : (j == 2) ? xyzw.z
                                : xyzw.w;
      }
    }
  }
  void ConstructorBLOCK(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0) ? x.x : (j == 1) ? y.y : (j == 2) ? z.z : w.w;
      }
    }
  }
  void ConstructorBLOCK(T x, T y, T z, T w) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0) ? x : (j == 1) ? y : (j == 2) ? z : w;
      }
    }
  }

  void ConstructorDIAGONAL(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        if (i == j) {
          mat4[i][j] = (i == 0) ? x.x : (i == 1) ? y.y : (i == 2) ? z.z : w.w;
        } else {
          mat4[i][j] = 0;
        }
      }
    }
  }
  void ConstructorDIAGONAL(Vec4<T> x) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        if (i == j) {
          mat4[i][j] = (i == 0) ? x.x : (i == 1) ? x.y : (i == 2) ? x.z : x.w;
        } else {
          mat4[i][j] = 0;
        }
      }
    }
  }
  void ConstructorDIAGONAL(T x, T y, T z, T w) {
    for (int8 i = 0; i < 4; i++) {
      for (int8 j = 0; j < 4; j++) {
        if (i == j) {
          mat4[i][j] = (i == 0) ? x : (i == 1) ? y : (i == 2) ? z : w;
        } else {
          mat4[i][j] = 0;
        }
      }
    }
  }
};

template <typename T = float> constexpr T lerp(T a, T b, T f) {
  return (a * (1.0f - f)) + (b * f);
}
template <typename T, typename S = float> constexpr T *lerp(T a, T b, S steps) {
  T *results = new T[steps + 1];

  float af = static_cast<float>(a);
  float bf = static_cast<float>(b);

  for (int32 n = 1; n <= steps; n++) {
    float stepSize = static_cast<float>(n) / static_cast<float>(steps);
    results[n - 1] = static_cast<T>(lerp(af, bf, stepSize));
  }
  return results;
}

template <typename T>
constexpr T *bezierCurveQUADRATIC(Vec3<T> v1, Vec3<T> v2, int steps) {
  T *results = new T[steps];

  for (float i = 0.f; i < static_cast<float>(steps); i++) {

    float t = i / static_cast<float>(steps);
    float xa = lerp(v1.x, v1.y, t);
    float ya = lerp(v2.x, v2.y, t);
    float xb = lerp(v1.y, v1.z, t);
    float yb = lerp(v2.y, v2.z, t);

    float x = lerp(xa, xb, t);
    float y = lerp(ya, yb, t);

    results[i] = {x, y};
  }

  return results;
}
template <typename T> constexpr T etc() {}
template <typename T = float> constexpr T *bhaskara(T a, T b, T c) {
  T *result = new T[2];
  T d = sqrt(pow(b, 2) - (4 * a * c));
  result[0] = (-b + d) / (2 * a);
  result[1] = (-b - d) / (2 * a);
  return result;
}
} // namespace Math
