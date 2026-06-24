// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/dev/vec_set.hpp>
//
namespace {
    using c10::ArrayRef;
    using ad_tensor::dev::vec_set_t;
    using ad_tensor::vector;
    //
    void check_empty(void) {
        vec_set_t        vec_set = vec_set_t();
        size_t           set_id  = vec_set.new_empty();
        ArrayRef<size_t> set     = vec_set.get_set(set_id);
        EXPECT_EQ(set.size(), 0);
    }
    //
    void check_singleton(void) {
        vec_set_t        vec_set = vec_set_t();
        size_t           set_id  = vec_set.new_singleton(5);
        ArrayRef<size_t> set     = vec_set.get_set(set_id);
        EXPECT_EQ(set.size(), 1);
        EXPECT_EQ(set[0],     5);
    }
    //
    void check_union(void) {
        vec_set_t        vec_set       = vec_set_t();
        vector<size_t> set_id;
        set_id.push_back( vec_set.new_singleton(5) );
        set_id.push_back( vec_set.new_singleton(4) );
        set_id.push_back( vec_set.new_empty() );
        vector<size_t> sets = {0, 1, 2};
        set_id.push_back( vec_set.union_sets(sets) );
        //
        for(size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(set_id[i], i);
        }
        c10::ArrayRef<size_t> union_sets = vec_set.get_set(3);
        EXPECT_EQ(union_sets.size(), 2);
        EXPECT_EQ(union_sets[0], 4);
        EXPECT_EQ(union_sets[1], 5);
    }
    //
    void check_link(void) {
        vec_set_t        vec_set       = vec_set_t();
        EXPECT_EQ(vec_set.data_size(), 0);
        vec_set.new_singleton(5);
        vec_set.new_singleton(4);
        EXPECT_EQ(vec_set.data_size(), 2);
        //
        // new set is not a link
        vec_set.union_sets({0, 1});
        EXPECT_EQ(vec_set.data_size(), 4);
        //
        // new set is a link
        size_t set_id_link = vec_set.union_sets({0, 1, 2});
        EXPECT_EQ(vec_set.data_size(), 5);
        //
        c10::ArrayRef<size_t> union_sets = vec_set.get_set(set_id_link);
        EXPECT_EQ(union_sets.size(), 2);
        EXPECT_EQ(union_sets[0], 4);
        EXPECT_EQ(union_sets[1], 5);
    }
}
TEST(tests_dev, vec_set)  {
    check_empty();
    check_singleton();
    check_union();
    check_link();
}
