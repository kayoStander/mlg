#pragma once

#include <cctype>
#include <cmath>
#include <cstring>

#include <stdint.h>

namespace {

template <typename T> class Vector {
public:
  virtual void sqrt() {};
  virtual void pow(const int32_t scalar) {};
  virtual void operator+(const T value) {};
  virtual void operator-(const T value) {};
  virtual void operator*(const T scalar) {};
  virtual void operator/(const T scalar) {};
  virtual T operator[](const int index) { return T(0.0); };
  virtual const T operator[](const int index) const { return T(0.0); };

private:
protected:
  virtual ~Vector(){};
};
template <typename T> class Matrix {
public:
  virtual void sqrt() {};
  virtual void pow(const int32_t scalar) {};
  virtual void operator+(const T value) {};
  virtual void operator-(const T value) {};
  virtual void operator*(const T scalar) {};
  virtual void operator/(const T scalar) {};

private:
  virtual void ConstructorDIAGONAL() {};
  virtual void ConstructorBLOCK() {};
  virtual void ConstructorVEC4() {};
};
} // namespace

namespace Math {

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
  Vec2(const T xy) : x(xy), y(xy) {}
  Vec2(const T x, const T y) : x(x), y(y) {}
  Vec2(const Vec2<T> &xy) : x(xy.x), y(xy.y) {}

  void pow(const int32_t scalar) {
    x = std::pow(static_cast<double>(x), scalar);
    y = std::pow(static_cast<double>(y), scalar);
  }
  void sqrt() {
    x = (x <= 0 ? 0 : static_cast<T>(std::sqrt(x)));
    y = (y <= 0 ? 0 : static_cast<T>(std::sqrt(y)));
  }
  void operator+(const T value) {
    x += value;
    y += value;
  }
  void operator-(const T value) {
    x -= value;
    y -= value;
  }
  void operator*(const T scalar) {
    x *= scalar;
    y *= scalar;
  }
  void operator/(const T scalar) {
    x /= scalar;
    y /= scalar;
  }

  T operator[](const int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    }
    return nullptr;
  }

  const T operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    }
    return nullptr;
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
  Vec3(const T xyz) : x(xyz), y(xyz), z(xyz) {}
  Vec3(const T x, const T y, const T z) : x(x), y(y), z(z) {}
  Vec3(const Vec3<T> &xyz) : x(xyz.x), y(xyz.y), z(xyz.z) {}
  Vec3(const Vec2<T> &xy, const T z) : x(xy.x), y(xy.y), z(z) {}
  Vec3(const T x, const Vec2<T> &yz) : x(x), y(yz.x), z(yz.y) {}

  void pow(const int32_t scalar) {
    x = std::pow(static_cast<double>(x), scalar);
    y = std::pow(static_cast<double>(y), scalar);
    z = std::pow(static_cast<double>(z), scalar);
  }
  void sqrt() {
    x = (x <= 0 ? 0 : static_cast<T>(std::sqrt(x)));
    y = (y <= 0 ? 0 : static_cast<T>(std::sqrt(y)));
    z = (z <= 0 ? 0 : static_cast<T>(std::sqrt(z)));
  }
  void operator+(const T value) {
    x += value;
    y += value;
    z += value;
  }
  void operator-(const T value) {
    x -= value;
    y -= value;
    z -= value;
  }
  void operator*(const T scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
  }
  void operator/(const T scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
  }

  T operator[](const int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    return nullptr;
  }

  const T operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    return nullptr;
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
  Vec4(const T xyzw) : x(xyzw), y(xyzw), z(xyzw), w(xyzw) {}
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

  void pow(const int32_t scalar) {
    x = std::pow(static_cast<double>(x), scalar);
    y = std::pow(static_cast<double>(y), scalar);
    z = std::pow(static_cast<double>(z), scalar);
    w = std::pow(static_cast<double>(w), scalar);
  }
  void sqrt() {
    x = (x <= 0 ? 0 : static_cast<T>(std::sqrt(x)));
    y = (y <= 0 ? 0 : static_cast<T>(std::sqrt(y)));
    z = (z <= 0 ? 0 : static_cast<T>(std::sqrt(z)));
    w = (w <= 0 ? 0 : static_cast<T>(std::sqrt(w)));
  }
  void operator+(const T value) {
    x += value;
    y += value;
    z += value;
    w += value;
  }
  void operator-(const T value) {
    x -= value;
    y -= value;
    z -= value;
    w -= value;
  }
  void operator*(const T scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    w *= scalar;
  }
  void operator/(const T scalar) {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    w /= scalar;
  }

  T operator[](const int index) {
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
  const T operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    case 3:
      return w;
    };
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

template <typename T> class Mat4x4 : public Matrix<T> {

  enum Constructor {
    MAT4X4DIAGONAL = 0,
    MAT4X4VEC4 = 1,
    MAT4X4BLOCK = 2,
  };

public:
  Mat4x4() { ConstructorBLOCK(); }
  Mat4x4(const T xyzw, const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(xyzw);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(xyzw);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(xyzw);
    };
  }
  Mat4x4(const T x, const T y, const T z, const T w,
         const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(x, y, z, w);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(x, y, z, w);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(x, y, z, w);
    };
  }
  Mat4x4(const Vec4<T> xyzw, const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(xyzw);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(xyzw);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(xyzw);
    };
  }
  Mat4x4(const Vec3<T> xyz, const T w,
         const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(xyz.x, xyz.y, xyz.z, w);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(xyz.x, xyz.y, xyz.z, w);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(xyz.x, xyz.y, xyz.z, w);
    };
  }
  Mat4x4(const Vec2<T> xy, const Vec2<T> zw,
         const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(xy.x, xy.y, zw.z, zw.w);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(xy.x, xy.y, zw.z, zw.w);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(xy.x, xy.y, zw.z, zw.w);
    };
  }
  Mat4x4(const Vec2<T> xy, const T z, const T w,
         const Constructor type = MAT4X4DIAGONAL) {
    switch (type) {
    case MAT4X4DIAGONAL:
      ConstructorDIAGONAL(xy.x, xy.y, z, w);
      break;
    case MAT4X4VEC4:
      ConstructorVEC4(xy.x, xy.y, z, w);
      break;
    case MAT4X4BLOCK:
      ConstructorBLOCK(xy.x, xy.y, z, w);
    };
  }

  void sqrt() {
    for (int8_t i = 0; i < 4; i++) {
      for (int8_t j = 0; j < 4; j++) {
        mat4[i][j] = (mat4[i][j] <= 0 ? 0 : std::sqrt(mat4[i][j]));
      }
    }
  };
  void pow(const int32_t scalar) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = std::pow(mat4[i][j], scalar);
      }
    }
  };
  void operator+(const T value) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] += value;
      }
    }
  };
  void operator-(const T value) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] -= value;
      }
    }
  };
  void operator*(const T scalar) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] *= scalar;
      }
    }
  };
  void operator/(const T scalar) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] /= scalar;
      }
    }
  };

  union {
    struct {
      T mat4[4][4];
    };
    struct {
      T xx, xy, xz, xw;
      T yx, yy, yz, yw;
      T zx, zy, zz, zw;
      T wx, wy, wz, ww;
    };
    struct {
      Vec4<T> &VecX;
      Vec4<T> &VecY;
      Vec4<T> &VecZ;
      Vec4<T> &VecW;
    };
  };

private:
  void ConstructorVEC4(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (uint8_t i = 0; i < 4; i++) {
      mat4[0][i] = x[i];
      mat4[1][i] = y[i];
      mat4[2][i] = z[i];
      mat4[3][i] = w[i];
    }
  }
  void ConstructorVEC4(Vec4<T> xyzw) {
    for (uint8_t i = 0; i < 4; i++) {
      mat4[0][i] = xyzw.x;
      mat4[1][i] = xyzw.y;
      mat4[2][i] = xyzw.z;
      mat4[3][i] = xyzw.w;
    }
  }
  void ConstructorVEC4(T x, T y, T z, T w) {
    for (uint8_t i = 0; i < 4; i++) {
      mat4[0][i] = x;
      mat4[1][i] = y;
      mat4[2][i] = z;
      mat4[3][i] = w;
    }
  }

  void ConstructorBLOCK() {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = 0;
      }
    }
  }
  void ConstructorBLOCK(Vec4<T> xyzw) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0)   ? xyzw.x
                     : (j == 1) ? xyzw.y
                     : (j == 2) ? xyzw.z
                                : xyzw.w;
      }
    }
  }
  void ConstructorBLOCK(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0) ? x.x : (j == 1) ? y.y : (j == 2) ? z.z : w.w;
      }
    }
  }
  void ConstructorBLOCK(T x, T y, T z, T w) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = (j == 0) ? x : (j == 1) ? y : (j == 2) ? z : w;
      }
    }
  }

  void ConstructorDIAGONAL(T xyzw) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat4[i][j] = (i == j) ? xyzw : 0;
      }
    }
  }
  void ConstructorDIAGONAL(Vec4<T> x, Vec4<T> y, Vec4<T> z, Vec4<T> w) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        if (i == j) {
          mat4[i][j] = (i == 0) ? x.x : (i == 1) ? y.y : (i == 2) ? z.z : w.w;
        } else {
          mat4[i][j] = 0;
        }
      }
    }
  }
  void ConstructorDIAGONAL(Vec4<T> x) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        if (i == j) {
          mat4[i][j] = (i == 0) ? x.x : (i == 1) ? x.y : (i == 2) ? x.z : x.w;
        } else {
          mat4[i][j] = 0;
        }
      }
    }
  }
  void ConstructorDIAGONAL(T x, T y, T z, T w) {
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
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

  for (uint32_t n = 1; n <= steps; n++) {
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
