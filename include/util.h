#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <vector>

namespace util {

std::string trim(const std::string& str, const std::string& whitespace = " \t") {
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos) return "";
  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}

void split(const std::string& s, char delim, std::vector<std::string>& elems) {
  elems.clear();
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

} // namespace util

#endif  // UTIL_H_