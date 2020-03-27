#pragma once

namespace cc {
namespace activation {

class function {
public:
    virtual float_t f(float_t v)  const = 0;
    virtual float_t df(float_t v) const = 0;
};

class tan_h : public function {
public:
    inline float_t f(float_t v) const override {
        return std::tanh(v);
    }
    inline float_t df(float_t v) const override {
        return 1.0f - sqr(v);
    }
};

class sigmoid : public function {
public:
    inline float_t f(float_t v) const override {
        return 1.0f / (1.0f + std::exp(-v));
    }
    inline float_t df(float_t v) const override {
        return v * (1.0f - v);
    }
};

class relu : public function {
public:
    inline float_t f(float_t v) const override {
        return std::max(v, float_t(0.f));
    }
    inline float_t df(float_t v) const override {
        return v > 0 ? 1.0f : 0.0f;
    }
};

class leaky_relu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0.f ? v : 0.01f * v;
    }
    inline float_t df(float_t v) const override {
        return v > 0.f ? 1.0f : 0.01f;
    }
};

#define K_PRELU_GRAD 0.2f
class parametric_relu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0.f ? v : K_PRELU_GRAD * v;
    }
    inline float_t df(float_t v) const override {
        return v > 0.f ? 1.0f : K_PRELU_GRAD;
    }
};

class elu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0.f ? v : std::exp(v) - 1.0f;
    }
    inline float_t df(float_t v) const override {
      return v > 0.f ? 1.0f : std::exp(v) + 1.0f;
    }
};

} // namespace activation
} // namespace cc
