// Copyright (C) 2013 IDP-G Team
// This file is part of IDP-G.
// 
// IDP-G is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// IDP-G is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with IDP-G.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __IDPG_ANSWER_DB__
#define __IDPG_ANSWER_DB__

#include <string>
#include <vector>

class AnswerDb {
public:
    AnswerDb();
    bool CheckAnswer(const std::string &method, int n, int seed, float answer);

private:
    struct Entry {
        int n;
        int seed;
        float answer;
        bool empty() {
            return n == -1;
        }
        Entry() {
            n = -1;
        }
    };
    std::vector<Entry> entries_;
    Entry Find(int n, int seed);
};

#endif
