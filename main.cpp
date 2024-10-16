#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cstring>
#include <unordered_map>

#include "Bitmap.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb/stb_image.h"

const uint32_t RED = 0x000000FF;
const uint32_t GREEN = 0x0000FF00;
const uint32_t BLUE = 0x00FF0000;


class Vec3f {
  public:
    float x, y, z;

    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    Vec3f(const Vec3f &v) : x(v.x), y(v.y), z(v.z) {}
    Vec3f operator*(const float &r) const { return Vec3f(x * r, y * r, z * r); }
    Vec3f operator/(const float &r) const { return Vec3f(x / r, y / r, z / r); }
    Vec3f operator*(const Vec3f &v) const { return Vec3f(x * v.x, y * v.y, z * v.z); }
    Vec3f operator-(const Vec3f &v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    Vec3f operator+(const Vec3f &v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    Vec3f &operator+=(const Vec3f &v) {
        x += v.x, y += v.y, z += v.z;
        return *this;
    }
};


float dotProduct(const Vec3f &a, const Vec3f &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float norma(const Vec3f &v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3f normalize(const Vec3f &v) {
    float mod = dotProduct(v, v);
    if (mod > 0) {
        return Vec3f(v/sqrt(mod));
    }
    return v;
}

bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else
    {
        x0 = (-b - sqrt(discr)) / (2 * a);
        x1 = (-b + sqrt(discr)) / (2 * a);
    }
    if (x0 > x1)
        std::swap(x0, x1);
    return true;
}

Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N * (2.f * dotProduct(N,I));
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index) {
    // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, dotProduct(I,N)));
    float etai = 1, etat = refractive_index;
    Vec3f n = N;
    if (cosi < 0) {
        cosi = -cosi;
        std::swap(etai, etat);
        n = N  * -1;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? Vec3f(0,0,0) : I * eta + n * (eta * cosi - sqrtf(k));
}


enum MaterialType
{
    DIFFUSE,
    GLOSSY,
    REFLECTION,
    REFLECTION_AND_REFRACTION
};

struct Material {
    Vec3f coefficients;
    Vec3f diffuse_color;
    MaterialType materialType;
    float specular;
    float refract;
    
    Material(const Vec3f &coeff, const Vec3f &color, const MaterialType &m, const float &s, const float &r) :
            coefficients(coeff), diffuse_color(color), materialType(m), specular(s), refract(r) {}
    Material() : coefficients(), diffuse_color(), materialType(DIFFUSE), specular(), refract() {}
};

class Object {
  public:
    Object() {}
    virtual ~Object() {}
    virtual void getData(const Vec3f &, Vec3f &, Material &) const = 0;
    virtual bool intersection(const Vec3f &, const Vec3f &, float &) const = 0;
};

class Sphere : public Object {
  public:
    Vec3f center;
    float radius;
    Material material;
    Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m){};

    bool intersection(const Vec3f &orig, const Vec3f &dir, float &x_near) const {
        Vec3f L = orig - center;
        float a = dotProduct(dir, dir);
        float b = 2 * dotProduct(dir, L);
        float c = dotProduct(L, L) - (radius * radius);
        float x0, x1;
        if (!solveQuadratic(a, b, c, x0, x1) || x1 < 0)
            return false;
        if (x0 < 0) {
            x_near = x1;
        } else {
            x_near = x0;
        }
        return true;
    }

    void getData(const Vec3f &hit_point, Vec3f &N, Material &m) const {
        m = material;
        N = normalize(hit_point - center);
    }
};


class Triangle : public Object {
    public:
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
    Material material;

    Triangle (const Vec3f &a,const Vec3f &b,const Vec3f &c, const Material &m) : v0(a), v1(b), v2(c), material(m) {}

    bool intersection(const Vec3f &orig, const Vec3f &dir, float &x_near) const {
        float a = v0.x - v1.x , b = v0.x - v2.x , c = dir.x , d = v0.x - orig.x;
        float e = v0.y - v1.y , f = v0.y - v2.y , g = dir.y , h = v0.y - orig.y;
        float i = v0.z - v1.z , j = v0.z - v2.z , k = dir.z , l = v0.z - orig.z;
        
        float m = f * k - g * j, q = g * i - e * k, s = e * j - f * i;
        float inv_denom = 1.0 / ( a * m + b * q + c * s);
        
        float n = h * k - g * l, p = f * l - h * j, r = e * l - h * i;
        float e1 = d * m - b * n - c * p;
        
        float beta = e1 * inv_denom;
        if(beta < 0.0)
            return false;
        float e2 = a * n + d * q + c * r;
        float gamma = e2 * inv_denom;
        if(gamma < 0.0 || beta + gamma > 1.0)
            return false;
        float e3 = a * p - b * r + d * s;
        float x = e3 * inv_denom;

        if (x < 1e-9)
            return false;
        x_near = x;
        return true;
    }


    void getData(const Vec3f &hit_point, Vec3f &N, Material &m) const {
        m = material;
        Vec3f va = v1 - v0, vb = v2 - v0;
        N = normalize(Vec3f(va.y * vb.z - va.z * vb.y, va.z * vb.x - va.x * vb.z, va.x * vb.y - va.y *  vb.x));
    }
};

class Plane : public Object {
    public:
    Vec3f v0, normal;
    Material material;
    Plane (const Vec3f &v, const Vec3f &n ,const Material &m) : v0(v), normal(n), material(m) {}

    bool intersection(const Vec3f &orig, const Vec3f &dir, float &x_near) const {
        float x = dotProduct((v0 - orig) , normal) / dotProduct(dir, normal);
        if (x < 1e-9)
            return false;
        x_near = x;
        return true;
    }

    void getData(const Vec3f &hit_point, Vec3f &N, Material &m) const {
        N = normalize(normal);
        m = material;
        if (dotProduct(N, Vec3f(0,0,-1))) {
            m.diffuse_color = (int(0.45 * hit_point.y + 1000)) & 1 ? Vec3f(1, 1, 1) : Vec3f(0, 0, 0);
        }
        else {
            m.diffuse_color = (int(0.3 * hit_point.x + 1000) + int(0.3 * hit_point.z)) & 1 ? Vec3f(1, 1, 1) : Vec3f(0, 0, 0);
            m.diffuse_color = m.diffuse_color * 0.5;
        }
    }
};



class Light {
  public:
    Light() {}
    virtual ~Light() {}
    virtual void get_Light(const Vec3f &, Vec3f &, Vec3f &, float &) const = 0;
};

class PointLight : public Light {
  public:
    Vec3f position;
    float intensity;
    Vec3f color;
    PointLight(const Vec3f &p, const float &i, const Vec3f &c) : position(p), intensity(i), color(c) {}
    void get_Light(const Vec3f &P, Vec3f &lightDir, Vec3f &lightIntensity, float &distance) const
    {
        lightDir = (position - P);
        distance = sqrt(norma(lightDir));
        
        lightDir = normalize(lightDir);
        lightIntensity = color * intensity;
    }
};


bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::vector<std::unique_ptr<Object>> &objects,
        Vec3f &hit, Vec3f &N, Material &material) {
    float objects_dist = std::numeric_limits<float>::max();
      
    for (size_t i = 0; i < objects.size(); ++i) {
        float near;
        if (objects[i]->intersection(orig, dir, near) && near < objects_dist) {
            objects_dist = near;
            hit = orig + dir * near;
            objects[i]->getData(hit, N, material);
        }
    }
    return objects_dist < 1000;
}



Vec3f TraceRay (const Vec3f &orig, const Vec3f &dir,
        const std::vector<std::unique_ptr<Object>> &objects,
        const std::vector<std::unique_ptr<Light>> &lights,
        const std::vector<Vec3f> &envmap,
        const int envmap_width,
        const int envmap_height,
        size_t depth = 0) {
    Vec3f Phong, hit_point, N;
    Material material;
    
    if (!(scene_intersect(orig, dir, objects, hit_point, N, material)) || depth > 5) {
        Sphere env(Vec3f(0, 0, 0), 1000, Material());
        float dist = 0;
        env.intersection(orig, dir, dist);
        Vec3f point = orig + dir * dist;
        int i = acos(point.y / 1000) / M_PI * envmap_height;
        int j = atan2(point.z, point.x) / (2 * M_PI) * envmap_width;
        return envmap[j + i * envmap_width];
        //return Vec3f();
    } else {
        
        switch(material.materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vec3f refract_dir = normalize(refract(dir, N, material.refract));
                Vec3f refract_orig = (dotProduct(refract_dir, N) > 0) ? hit_point + N * 1e-3 : hit_point - N * 1e-3;
                Vec3f refract_color = TraceRay(refract_orig, refract_dir, objects, lights, envmap, envmap_width, envmap_height, depth + 1);
                Phong = refract_color * 0.8;
            }
            case REFLECTION:
            case GLOSSY:
            {
                Vec3f reflect_dir = reflect(dir, N);
                Vec3f reflect_orig = (dotProduct(reflect_dir, N) > 0) ? hit_point + N * 1e-3 : hit_point - N * 1e-3;
                Vec3f reflect_color = TraceRay(reflect_orig, reflect_dir, objects, lights, envmap, envmap_width, envmap_height, depth + 1);
                Phong += reflect_color * material.coefficients.z;
            }
            case DIFFUSE:
            {
                Vec3f diffuse(0, 0, 0), specular(0, 0, 0);
                
                for (uint32_t i = 0; i < lights.size(); ++i) {
                    Vec3f light_dir, light_intensity;
                    float light_dist;
                    lights[i]->get_Light(hit_point, light_dir, light_intensity, light_dist);
                    
                    //shadows
                    Material tmp;
                    Vec3f shadow_pt, shadow_N;
                    Vec3f shadow_orig = dotProduct(light_dir, N) > 0 ? hit_point + N*1e-3 : hit_point - N*1e-3;
                    if (scene_intersect(shadow_orig, light_dir, objects, shadow_pt, shadow_N, tmp) && norma((shadow_pt-shadow_orig)) < light_dist)
                        continue;
                                   
                    diffuse += light_intensity * std::max(0.f, dotProduct(light_dir, N));
                    specular += light_intensity * powf(std::max(0.f, dotProduct(reflect(light_dir, N), dir)), material.specular);
                    
                    
                }
                Phong += material.diffuse_color * diffuse * material.coefficients.x + specular * material.coefficients.y;
                break;
            } 
        }
    }
    
    return Phong;
}

int main(int argc, const char **argv) {
    std::unordered_map<std::string, std::string> cmdLineParams;

    for(int i=0; i<argc; i++) {
        std::string key(argv[i]);
        if(key.size() > 0 && key[0] == '-') {
            if(i != argc-1) {
                cmdLineParams[key] = argv[i + 1];
                i++;
            }
            else
        cmdLineParams[key] = "";
        }
    }
    std::string outFilePath = "zout.bmp";
    if(cmdLineParams.find("-out") != cmdLineParams.end())
        outFilePath = cmdLineParams["-out"];
    int sceneId = 0, threads = 0;
    if(cmdLineParams.find("-scene") != cmdLineParams.end())
        sceneId = atoi(cmdLineParams["-scene"].c_str());
    if(cmdLineParams.find("-threads") != cmdLineParams.end())
        threads = atoi(cmdLineParams["-threads"].c_str());
    
    int width = 1500;
    int height = 1500;
    float fov = 90;
    float AA = 4;
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<std::unique_ptr<Light>> lights;
    
    lights.push_back(std::unique_ptr<Light>(new PointLight(Vec3f(-20, -2, 10), 1.5, Vec3f(1, 1, 1))));
    lights.push_back(std::unique_ptr<Light>(new PointLight(Vec3f( 30, 50,  -25), 1.8, Vec3f(1, 1, 1))));
    lights.push_back(std::unique_ptr<Light>(new PointLight(Vec3f(30, 20,  30), 1.7, Vec3f(1, 1, 1))));
    
    Material red(Vec3f(0.9, 0.2, 0.3), Vec3f(0.3, 0.0, 0.0), DIFFUSE, 10, 1.0);
    Material swamp(Vec3f(0.9, 0.2, 0.3), Vec3f(0.44, 0.4, 0.11), DIFFUSE, 10, 1.0);
    Material blue(Vec3f(0.5, 0.1, 0.7), Vec3f(0.0, 0.1, 0.4), GLOSSY, 5, 1.0);
    Material mirror(Vec3f(0.0, 1.0, 0.85), Vec3f(1.0, 1.0, 1.0), REFLECTION, 1024, 1.0);
    Material glass(Vec3f(0.0,  0.3, 0.1), Vec3f(0.6, 0.7, 0.8), REFLECTION_AND_REFRACTION, 100, 1.5);
    Material checker(Vec3f(0.5,  0.35, 0.5), Vec3f(0.0, 0.0, 0.0), GLOSSY, 5, 1.0);
    
    
    Vec3f ta = Vec3f(9, -3, -12);
    Vec3f tb = Vec3f(13, -3, -16);
    Vec3f tc = Vec3f(7, -3, -16);
    Vec3f top = Vec3f(9, 3, -14);
    objects.push_back(std::unique_ptr<Object>(new Triangle(ta, top, tc, blue)));
    objects.push_back(std::unique_ptr<Object>(new Triangle(ta, tb, top, blue)));
    objects.push_back(std::unique_ptr<Object>(new Triangle(tc, tb, top, blue)));
    objects.push_back(std::unique_ptr<Object>(new Triangle(ta, tb, tc, blue)));
    
    objects.push_back(std::unique_ptr<Object>(new Sphere(Vec3f(-6, 0, -10), 2, swamp)));
    objects.push_back(std::unique_ptr<Object>(new Sphere(Vec3f(5, 5, -20), 4, mirror)));
    objects.push_back(std::unique_ptr<Object>(new Sphere(Vec3f(-2.5, -0.5, -16), 3, glass)));
    objects.push_back(std::unique_ptr<Object>(new Plane(Vec3f(0, -4, 0), Vec3f(0, 1, 0), checker)));


    std::vector<Vec3f> envmap;
    std::vector<uint32_t> image(height * width * 3);
    
    
    int envmap_width, envmap_height, n = 0;
    unsigned char *map = stbi_load("./envmap.jpg", &envmap_width, &envmap_height, &n, 0);
    if (!map) {
        std::cerr << "Error: can not load the environment map" << std::endl;
        return -1;
    }
    envmap = std::vector<Vec3f>(envmap_width * envmap_height);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < envmap_width; ++i) {
        for (int j = 0; j < envmap_height; ++j) {
            envmap[j + i * envmap_height] = Vec3f(map[(j + i * envmap_height) * 3 + 0],
                                                  map[(j + i * envmap_height) * 3 + 1],
                                                  map[(j + i * envmap_height) * 3 + 2]) / 255;
        }
    }
    stbi_image_free(map);
    
    
    float scale = tan(fov * 0.5 * M_PI / 180);
    float AspectRatio = width / (float)height;
    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
            Vec3f color;
            
            for (size_t k = 0; k < AA; ++k) {
                float x = (2 * (j + 0.5 + k * 0.25) / (float)width - 1) * scale * AspectRatio;
                float y = (2 * (i + 0.5 - k * 0.25) / (float)height - 1) * scale;
                Vec3f dir = normalize(Vec3f(x, y, -1));
                color += TraceRay(Vec3f(0, 0, 0), dir, objects, lights, envmap, envmap_width, envmap_height);
            }
            color = color / AA;
            
            
            float max = std::max(color.x, std::max(color.y, color.z));
            if (max > 1)
                color = color / max;
            image[j + i * height] = (uint32_t)(255 * std::max(0.f, std::min(1.f, color.z))) << 16 |
                                   (uint32_t)(255 * std::max(0.f, std::min(1.f, color.y))) << 8 |
                                   (uint32_t)(255 * std::max(0.f, std::min(1.f, color.x)));
        }
    }
    
    SaveBMP(outFilePath.c_str(), image.data(), width, height);
    return 0;
}
