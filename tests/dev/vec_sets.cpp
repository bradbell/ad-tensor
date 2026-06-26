// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/dev/vec_sets.hpp>
//
namespace {
    using c10::ArrayRef;
    using ad_tensor::dev::vec_sets_t;
    using ad_tensor::vector;
    //
    void check_empty(void) {
        vec_sets_t       vec_sets= vec_sets_t();
        size_t           set_id  = vec_sets.empty_set();
        ArrayRef<size_t> set     = vec_sets.get_set(set_id);
        EXPECT_EQ(set.size(), 0);
        //
        EXPECT_EQ(vec_sets.n_set(), 1);
    }
    //
    void check_singleton(void) {
        vec_sets_t       vec_sets = vec_sets_t();
        size_t           set_id   = vec_sets.singleton_set(5);
        ArrayRef<size_t> set      = vec_sets.get_set(set_id);
        EXPECT_EQ(set.size(), 1);
        EXPECT_EQ(set[0],     5);
        //
        EXPECT_EQ(vec_sets.n_set(), 1);
    }
    //
    void check_union(void) {
        vec_sets_t       vec_sets      = vec_sets_t();
        vector<size_t> set_id;
        set_id.push_back( vec_sets.singleton_set(5) );
        set_id.push_back( vec_sets.singleton_set(4) );
        set_id.push_back( vec_sets.empty_set() );
        vector<size_t> sub_sets = {0, 1, 2};
        set_id.push_back( vec_sets.union_set(sub_sets) );
        //
        for(size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(set_id[i], i);
        }
        c10::ArrayRef<size_t> union_set = vec_sets.get_set(3);
        EXPECT_EQ(union_set.size(), 2);
        EXPECT_EQ(union_set[0], 4);
        EXPECT_EQ(union_set[1], 5);
        //
        EXPECT_EQ(vec_sets.n_set(), 4);
    }
    //
    void check_link(void) {
        vec_sets_t       vec_sets      = vec_sets_t();
        EXPECT_EQ(vec_sets.n_data(), 0);
        vec_sets.singleton_set(5);
        vec_sets.singleton_set(4);
        EXPECT_EQ(vec_sets.n_data(), 2);
        //
        // new set is not a link
        vec_sets.union_set({0, 1});
        EXPECT_EQ(vec_sets.n_data(), 4);
        //
        // new set is a link
        size_t set_id_link = vec_sets.union_set({0, 1, 2});
        EXPECT_EQ(vec_sets.n_data(), 5);
        //
        c10::ArrayRef<size_t> union_set = vec_sets.get_set(set_id_link);
        EXPECT_EQ(union_set.size(), 2);
        EXPECT_EQ(union_set[0], 4);
        EXPECT_EQ(union_set[1], 5);
        //
        EXPECT_EQ(vec_sets.n_set(), 4);
    }
}
TEST(tests_dev, vec_sets)  {
    check_empty();
    check_singleton();
    check_union();
    check_link();
}
