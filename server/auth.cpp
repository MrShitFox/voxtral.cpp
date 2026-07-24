#include "auth.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace voxtral::server {

bool constant_time_equal(std::string_view left, std::string_view right) {
    const std::size_t length = std::max(left.size(), right.size());
    std::uint64_t difference =
        static_cast<std::uint64_t>(left.size() ^ right.size());
    for (std::size_t i = 0; i < length; ++i) {
        const unsigned char left_byte =
            i < left.size() ? static_cast<unsigned char>(left[i]) : 0;
        const unsigned char right_byte =
            i < right.size() ? static_cast<unsigned char>(right[i]) : 0;
        difference |= static_cast<std::uint64_t>(left_byte ^ right_byte);
    }
    return difference == 0;
}

bool authorize_bearer(
    std::string_view authorization_header,
    std::string_view configured_key,
    bool authentication_enabled)
{
    if (!authentication_enabled) {
        return true;
    }
    constexpr std::string_view prefix = "Bearer ";
    if (authorization_header.size() <= prefix.size() ||
        authorization_header.substr(0, prefix.size()) != prefix) {
        return false;
    }
    const std::string_view token = authorization_header.substr(prefix.size());
    if (token.empty() || token.front() == ' ' || token.back() == ' ' ||
        token.front() == '\t' || token.back() == '\t') {
        return false;
    }
    return constant_time_equal(token, configured_key);
}

} // namespace voxtral::server
