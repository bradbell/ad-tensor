#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<string>

namespace ad_tensor {
    class options_t {
    private:
        std::string m_name;
        bool        m_trace;
    public:
        // ctor
        options_t(void);
        //
        // name
        void set_name(const std::string& name);
        const std::string& get_name(void) const;
        //
        // trace
        void set_trace(bool trace);
        bool get_trace(void) const;
    };
}
