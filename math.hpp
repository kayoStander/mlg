#pragma once

/////////////////////////////////

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <type_traits>
#include <utility>

#if (MLGDOCUMENT)
#include "mlg.document.hpp"
#endif

/////////////////////////////////

namespace {

template <typename T, typename = void> struct HasMemberMat : std::false_type {};
template <typename T, typename = void> struct HasMemberX : std::false_type {};
template <typename T>
struct HasMemberMat<T, std::void_t<decltype(std::declval<T>().mat)>>
    : std::true_type {};
template <typename T>
struct HasMemberX<T, std::void_t<decltype(std::declval<T>().x)>>
    : std::true_type {};

template <class T> class Vector {
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
template <class T> class Matrix {
public:
  virtual void sqrt() {};
  virtual void pow(const int32_t scalar) {};
  virtual void operator+(const T value) {};
  virtual void operator-(const T value) {};
  virtual void operator*(const T scalar) {};
  virtual void operator/(const T scalar) {};

  virtual T Determinant() { return T(0.0); };
  virtual const T Determinant() const { return T(0.0); };

private:
protected:
  template <typename... Args, typename Mat>
  constexpr void ConstructorDiagonal(Mat &Matrix, Args &&...args) {
    static_assert(HasMemberMat<Mat>{}, "1st value isn't a Matrix");
    static_assert(std::extent<decltype(Matrix.mat)>::value !=
                      std::extent<decltype(Matrix.mat[0])>::value,
                  "ConstructorDiagonal called on not same sized Matrix "
                  "(2x2,3x3,4x4;)");
    for (size_t i = 0; i < std::extent<decltype(Matrix.mat)>::value; i++) {
      ((Matrix.mat[i][i] = std::forward<Args>(args)), ...);
    }
  };
  template <typename... Args, typename Mat>
  constexpr void ConstructorBlock(Mat &Matrix, Args &&...args) {
    static_assert(HasMemberMat<Mat>{}, "1st value isn't a Matrix");
    if (sizeof...(args) > 4 && !HasMemberX<decltype(((args), ...))>{} &&
        !HasMemberMat<decltype(((args), ...))>{}) {
      for (size_t i = 0; i < std::extent<decltype(Matrix.mat)>::value; i++) {
        // float,int,double && args > 4
        size_t j = 0;
        ((Matrix.mat[0][j++] = std::forward<Args>(args)), ...);
      }
      return;
    } else if (sizeof...(args) > 4 && HasMemberX<decltype(((args), ...))>{} &&
               !HasMemberMat<decltype(((args), ...))>{}) {
      for (size_t i = 0; i < std::extent<decltype(Matrix.mat)>::value; i++) {
        // vec2,vec3,vec4 && args > 4
        size_t j = 0;
        // FIX Matrix.mat[0][j++] = ((std::forward<Args>(args)), ...)[j];
        return;
      }
    } else if (sizeof...(args) > 4 && !HasMemberX<decltype(((args), ...))>{} &&
               HasMemberMat<decltype(((args), ...))>{}) {
      // matrix2x2,matrix3x3,matrix4x4 && args > 4
      // ...
      return;
    }
    // args <= 4;
    for (size_t i = 0; i < std::extent<decltype(Matrix.mat)>::value; i++) {
      size_t j = 0;
      ((Matrix.mat[i][j++] = std::forward<Args>(args)), ...);
    }
  };
  template <typename... Args, typename Mat>
  constexpr void ConstructorVector(Mat &Matrix, Args &&...args) {
    static_assert(HasMemberMat<Mat>{}, "1st value isn't a Matrix");
    static_assert(!HasMemberX<decltype(((args), ...))>{},
                  "Arguments aren't have vectors");
    for (size_t i = 0; i < std::extent<decltype(Matrix.mat)>::value; i++) {
      size_t j = 0;
      Matrix.mat[i][j++] = ((std::forward<Args>(args)), ...)[j];
    }
  };
};
} // namespace

namespace Mlg {

enum Constructor {
  MATRIXDIAGONAL = 0,
  MATRIXVECTOR = 1,
  MATRIXBLOCK = 2,
};

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

template <class T> class Vec2 : public Vector<T> {
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
    default:
      throw std::runtime_error("Index out of range");
    }
  }

  const T operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    default:
      throw std::runtime_error("Index out of range");
    }
  }

public:
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

template <class T> class Vec3 : public Vector<T> {
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
    default:
      throw std::runtime_error("Index out of range");
    }
  }

  const T operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    default:
      throw std::runtime_error("Index out of range");
    }
  }

public:
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

template <class T> class Vec4 : public Vector<T> {
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
    default:
      throw std::runtime_error("Index out of range");
    }
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
    default:
      throw std::runtime_error("Index out of range");
    };
  }

public:
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
template <class T> class Matrix3x3 : public Matrix<T> { // TODO
public:
  static_assert(
      std::is_arithmetic<T>::value || std::is_same<T, Vec2<T>>::value ||
          std::is_same<T, Vec3<T>>::value || std::is_same<T, Vec4<T>>::value ||
          std::is_same<T, Matrix3x3<T>>::value,
      "Matrix only accepts numbers/vectors/matrix(with lower/same size)");

  template <typename... Args> Matrix3x3(Args &&...args) {
    this->template ConstructorBlock(*this, (args)...);
  }

  void sqrt() {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        mat[i][i] = (mat[i][i] <= 1 ? mat[i][i] : std::sqrt(mat[i][i]));
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        mat[i][j] = (mat[i][j] <= 1 ? mat[i][j] : std::sqrt(mat[i][j]));
      }
    }
  };
  void pow(const int32_t scalar) {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t k = 0; k < scalar; k++) {
          mat[i][i] *= mat[i][i];
        }
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        for (uint8_t k = 0; k < scalar; k++) {
          mat[i][j] *= mat[i][j];
        }
      }
    }
  };
  void operator+(const T value) {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        mat[i][i] += value;
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        mat[i][j] += value;
      }
    }
  };
  void operator-(const T value) {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        mat[i][i] -= value;
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        mat[i][j] -= value;
      }
    }
  };
  void operator*(const T scalar) {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        mat[i][i] *= scalar;
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        mat[i][j] *= scalar;
      }
    }
  };
  void operator/(const T scalar) {
    if (world) {
      for (uint8_t i = 0; i < 3; i++) {
        mat[i][i] /= scalar;
      }
      return;
    }
    for (uint8_t i = 0; i < 3; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        mat[i][j] /= scalar;
      }
    }
  };

  T Determinant() { return T(0.0); }

public:
  union {
    struct {
      T mat[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    };
    struct {
      T xx, xy, xz;
      T yx, yy, yz;
      T zx, zy, zz;
    };
    struct {
      Vec3<T> &VecX;
      Vec3<T> &VecY;
      Vec3<T> &VecZ;
    };
  };

  bool world = true;

private:
};
template <class T> class Matrix4x4 : public Matrix<T> {

public:
  static_assert(std::is_arithmetic<T>::value ||
                    std::is_same<T, Vec2<T>>::value ||
                    std::is_same<T, Vec3<T>>::value ||
                    std::is_same<T, Vec4<T>>::value ||
                    std::is_same<T, Matrix3x3<T>>::value ||
                    std::is_same<T, Matrix4x4<T>>::value,
                "Matrix only accepts numbers/vectors/matrix");

  /*
   * Matrix4x4()
   * @param TODO
   * @note TODO
   */
  template <typename... Args> Matrix4x4(Args &&...args) {
    this->template ConstructorBlock(*this, (args)...);
  }
  void sqrt() {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        mat[i][i] = (mat[i][i] <= 1 ? mat[i][i] : std::sqrt(mat[i][i]));
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat[i][j] = (mat[i][j] <= 1 ? mat[i][j] : std::sqrt(mat[i][j]));
      }
    }
  };
  void pow(const int32_t scalar) {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        for (uint32_t k = 0; k < scalar; k++) {
          mat[i][i] *= mat[i][i];
        }
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        for (uint32_t k = 0; k < scalar; k++) {
          mat[i][i] *= mat[i][i];
        }
      }
    }
  };
  void operator+(const T value) {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        mat[i][i] += value;
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat[i][j] += value;
      }
    }
  };
  void operator-(const T value) {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        mat[i][i] -= value;
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat[i][j] -= value;
      }
    }
  };
  void operator*(const T scalar) {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        mat[i][i] *= scalar;
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat[i][j] *= scalar;
      }
    }
  };
  void operator/(const T scalar) {
    if (world) {
      for (uint8_t i = 0; i < 4; i++) {
        mat[i][i] /= scalar;
      }
      return;
    }
    for (uint8_t i = 0; i < 4; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        mat[i][j] /= scalar;
      }
    }
  };

  T Determinant() { return T(0.0); }

  /*template <
      typename = std::enable_if<std::is_same<Matrix4x4<T>, T>::value, void>>
  void temp() {}*/

public:
  union {
    struct {
      T mat[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
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

  bool world = true;

private:
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
} // namespace Mlg
