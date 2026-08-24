// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
//
namespace {
    //
    // vector
    using ad_tensor::vector;
    //
    // write_vec_at_ten
    void write_vec_at_ten(
        const vector<at::Tensor> vec_at_ten ,
        const std::string&       file_name  ) {
        //
        // file
        std::ofstream file(file_name, std::ios::binary);
        //
        // n_ten
        size_t n_ten          = vec_at_ten.size();
        const char* n_ten_ptr = reinterpret_cast<const char*>(&n_ten);
        file.write(n_ten_ptr , sizeof(size_t) );
        //
        for(size_t i = 0; i < n_ten; ++i) {
            //
            // ten_i
            at::Tensor ten_i = vec_at_ten[i].contiguous().cpu();
            //
            //
            // type
            int64_t type         = static_cast<int64_t>( ten_i.scalar_type() );
            const char* type_ptr = reinterpret_cast<const char*>(&type);
            file.write(type_ptr, sizeof(int64_t) );
            //
            // n_dim
            int64_t n_dim         = ten_i.dim();
            const char* n_dim_ptr = reinterpret_cast<const char*>(&n_dim);
            file.write(n_dim_ptr, sizeof(int64_t) );
            //
            // shape
            const char* shape_ptr =
                reinterpret_cast<const char*>( ten_i.sizes().data() );
            file.write(shape_ptr, n_dim * sizeof(int64_t) );
            //
            // data
            int64_t n_elements   = ten_i.numel();
            int64_t n_bytes      = n_elements * ten_i.element_size();
            const char* data_ptr =
                reinterpret_cast<const char*>( ten_i.data_ptr() );
            file.write(data_ptr, n_bytes);
        }
    }
    //
    // read_vec_at_ten
    vector<at::Tensor> read_vec_at_ten(const std::string& file_name ) {
        //
        // file
        std::ifstream file(file_name, std::ios::binary);
        //
        // n_ten
        size_t n_ten;
        char* n_ten_ptr = reinterpret_cast<char*>(&n_ten);
        file.read(n_ten_ptr , sizeof(size_t) );
        //
        // vec_at_ten
        vector<at::Tensor> vec_at_ten(n_ten);
        //
        for(size_t i = 0; i < n_ten; ++i) {
            //
            // scalar_type
            int64_t type;
            char* type_ptr = reinterpret_cast<char*>(&type);
            file.read(type_ptr, sizeof(int64_t) );
            auto scalar_type = static_cast<torch::ScalarType>(type);
            //
            // n_dim
            int64_t n_dim;
            char* n_dim_ptr = reinterpret_cast<char*>(&n_dim);
            file.read(n_dim_ptr, sizeof(int64_t) );
            //
            // shape
            vector<int64_t> shape(n_dim);
            char* shape_ptr = reinterpret_cast<char*>( shape.data() );
            file.read(shape_ptr, n_dim * sizeof(int64_t) );
            //
            // ten_i
            at::Tensor ten_i = torch::empty(shape, torch::dtype(scalar_type) );
            int64_t n_bytes  = ten_i.numel() * ten_i.element_size();
            char* data_ptr   = reinterpret_cast<char*>( ten_i.data_ptr() );
            file.read(data_ptr, n_bytes);
            //
            // vec_at_ten
            vec_at_ten[i] = ten_i;
        }
        return vec_at_ten;
    }
}
//
TEST(tests, fileio) {
    using std::filesystem::path;
    //
    // temp_dir
    path temp_dir = std::filesystem::temp_directory_path();
    //
    // tests_fileio_path
    path tests_fileio_path = temp_dir / "tests_fileio.binary";
    //
    // file_name
    std::string file_name = tests_fileio_path.string();
    //
    // vec_at_ten
    vector<at::Tensor> vec_at_ten_in;
    vec_at_ten_in.push_back( torch::tensor( { {1, 2}, {3, 4} } ) );
    vec_at_ten_in.push_back( torch::tensor( { 5.0, 6.0, 7.0, 8.0 } ) );
    //
    // tests_filio_path
    write_vec_at_ten(vec_at_ten_in, file_name);
    //
    // vec_at_ten_output
    vector<at::Tensor> vec_at_ten_out = read_vec_at_ten(file_name);
    //
    EXPECT_EQ(vec_at_ten_out.size(), 2);
    EXPECT_TRUE( vec_at_ten_out[0].equal( vec_at_ten_in[0] ) );
    EXPECT_TRUE( vec_at_ten_out[1].equal( vec_at_ten_in[1] ) );
}
