#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin options usr}

The Options Key Value Map
#########################

trace
*****
{xrst_literal ,
    BEGIN_TRACE, END_TRACE
}
The default value for trace is false.

{xrst_end options}
*/
#include<map>
#include<string>

namespace ad_tensor {
    class options_t {
    private:
        bool m_trace;
    public:
        options_t(void)
        : m_trace(false)
        { }
        //
        // BEGIN_TRACE
        void set_trace(bool trace);
        bool get_trace(void) const;
        // END_TRACE
    };
}
