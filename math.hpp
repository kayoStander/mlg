#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {
class Vector {
public:

  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;
  Vector(const Vector &&) = delete;
  Vector &operator=(const Vector &&) = delete;

  virtual void etc() {}

private:
};
class Matrix {
public:

  Matrix(const Matrix &) = delete;
  Matrix &operator=(const Matrix &) = delete;
  Matrix(const Matrix &&) = delete;
  Matrix &operator=(const Matrix &&) = delete;

  virtual void etc() {}

private:
};
} // namespace

namespace Math {
enum class BEZIERTYPE { QUADRADIC = 0, CUBIC = 1, LINEAR = 2 };

namespace Constant {
template <typename T> constexpr static T PI() {
  return static_cast<T>(3.1415926535f);
}
template <typename T> constexpr static T HALFPI() {
  return static_cast<T>(1.5707963267f);
}
template <typename T> constexpr static T ONEOVERPI() {
  return static_cast<T>(0.3183098861f);
}

template <typename T> constexpr static T EULER() {
  return static_cast<T>(0.5772156649f);
}
template <typename T> constexpr static T GOLDENRATIO() {
  return static_cast<T>(1.618f);
}
} // namespace Constant

template <typename T> class Vec2 : Vector {
public:
  Vec2(T x, T y) : x(x), y(y) {}
  T x, y;

private:
};

template <typename T> class Vec3 : Vector {
public:
  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  T x, y, z;

private:
};

template <typename T> class Mat4x4 : Matrix {
public:
private:
};

template <typename T = float> static constexpr T lerp(T a, T b, T f) {
  return (a * (1.0f - f)) + (b * f);
}
template <typename T, typename S = float>
constexpr static T *lerp(T a, T b, S steps) {
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
static void bezierCurve(Vec3<T> v1, Vec3<T> v2, int steps,
                        BEZIERTYPE type = BEZIERTYPE::QUADRADIC) {
  for (float i = 0.f; i < static_cast<float>(steps); i++) {

    float t = i / static_cast<float>(steps);
    float xa = lerp(v1.x, v1.y, t);
    float ya = lerp(v2.x, v2.y, t);
    float xb = lerp(v1.y, v1.z, t);
    float yb = lerp(v2.y, v2.z, t);

    float x = lerp(xa, xb, t);
    float y = lerp(ya, yb, t);

    std::cout << x << " // " << y << std::endl;
  }
}
static float *bhaskara(float a, float b, float c) {
  float *result = new float[2];
  float d = sqrt(pow(b, 2) - (4 * a * c));
  result[0] = (-b + d) / (2 * a);
  result[1] = (-b - d) / (2 * a);
  return result;
}
} // namespace Math
