#include "nbody.h"
#include "utility.cuh"
#include <random>

//disregard this, the lambda madness isn't worth it and won't compile properly


thrust::device_vector<float4> rand_pos_mass(size_t N)
{
    thrust::device_vector<float4> ret;

    std::default_random_engine generator(0xdeadbeef);
    std::uniform_real_distribution<float> position_distribution(-1,1);
    auto x = std::bind(position_distribution, generator);
    std::uniform_real_distribution<float> mass_distribution(0.1f,5.f);
    auto m = std::bind(mass_distribution, generator);

    for(auto _=0;_!=N;++_) {
        ret.push_back({x(),x(),x(),m()});
    }
    return ret;
}

seconds nbody_thrust(size_t N, size_t iters)
{
    thrust::device_vector<float4> pos_mass = rand_pos_mass(N);
    thrust::device_vector<float3> velocities(N);
    thrust::fill(velocities.begin(), velocities.end(), float3{0.f, 0.f, 0.f});

    auto make_vel_update = [](float4 current, float step){
    return [&current,step](float4 const& other){
        float3 diff = {current.x - other.x,
                       current.y - other.y,
                       current.z - other.z};
        float3 update = dt/step * G * current.w * diff / (norm_pow3(diff) + eps);
        return float3{update.x, update.y, update.z};
    };
    };

    auto pos_update = [](float3 velocity, float4 pos_mass){
        return float4{
            pos_mass.x + velocity.x * dt,
            pos_mass.y + velocity.y * dt,
            pos_mass.z + velocity.z * dt
        };
    };

    auto start = std::chrono::high_resolution_clock::now();

    // fist halfstep velocity update
    thrust::device_vector<float3> dv;
    for(auto current: pos_mass)
    {
        auto updates = thrust::make_transform_iterator(pos_mass.begin(), make_vel_update(current, 2));
        auto updates_end = thrust::make_transform_iterator(pos_mass.end(), make_vel_update(current, 2));

        dv.push_back(thrust::reduce(updates,updates_end,
                float3{0.f,0.f,0.f}));
    }
    thrust::transform(dv.begin(),dv.end(),velocities.begin(),thrust::plus<float>());

    for(auto _ = 0; _!= iters; ++_) {
        // velocity update
        for (auto current: pos_mass) {
            auto updates = thrust::make_transform_iterator(pos_mass.begin(), make_vel_update(current, 1));
            auto updates_end = thrust::make_transform_iterator(pos_mass.end(), make_vel_update(current, 1));

            dv.push_back(thrust::reduce(updates, updates_end));
        }
        thrust::transform(dv.begin(), dv.end(), velocities.begin(), thrust::plus<float>());

        //position update

        thrust::transform(velocities.begin(),velocities.end(), pos_mass.begin(), pos_mass.begin(),pos_update);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<seconds>(end - start);
}
