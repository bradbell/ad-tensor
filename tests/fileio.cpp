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
    // save_ad_vec_at_ten
    void save_ad_vec_at_ten(
        const ad_tensor::vector<at::Tensor>& ad_vec_at_ten ,
        const std::string&                   file_name  ) {
        //
        // std_vec_at_ten_ptr
        auto std_vec_at_ten_ptr =
            dynamic_cast< const std::vector<at::Tensor>* >(&ad_vec_at_ten);
        //
        // file_name file
        torch::save(*std_vec_at_ten_ptr, file_name);
    }
    //
    // load_ad_vec_at_ten
    ad_tensor::vector<at::Tensor> load_ad_vec_at_ten(
        const std::string& file_name ) {
        //
        // ad_vec_at_ten
        ad_tensor::vector<at::Tensor> ad_vec_at_ten;
        //
        // std_vec_at_ten_ptr
        auto std_vec_at_ten_ptr =
            dynamic_cast< std::vector<at::Tensor>* >(&ad_vec_at_ten);
        //
        // ad_vec_at_ten
        torch::load(*std_vec_at_ten_ptr, file_name);
        //
        return ad_vec_at_ten;
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
    ad_tensor::vector<at::Tensor> ad_vec_at_ten_in;
    ad_vec_at_ten_in.push_back( torch::empty( {0} ) );
    ad_vec_at_ten_in.push_back( torch::tensor( { {1, 2}, {3, 4} } ) );
    ad_vec_at_ten_in.push_back( torch::tensor( { 5.0, 6.0, 7.0, 8.0 } ) );
    //
    // file_name file
    save_ad_vec_at_ten(ad_vec_at_ten_in, file_name);
    //
    // ad_vec_at_ten_output
    ad_tensor::vector<at::Tensor> ad_vec_at_ten_out =
        load_ad_vec_at_ten(file_name);
    //
    EXPECT_EQ(ad_vec_at_ten_out.size(), 3);
    EXPECT_TRUE( ad_vec_at_ten_out[0].equal( ad_vec_at_ten_in[0] ) );
    EXPECT_TRUE( ad_vec_at_ten_out[1].equal( ad_vec_at_ten_in[1] ) );
    EXPECT_TRUE( ad_vec_at_ten_out[2].equal( ad_vec_at_ten_in[2] ) );
}
