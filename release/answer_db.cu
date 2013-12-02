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

#include "answer_db.cuh"
#include "common.cuh"
#include <cassert>
using namespace std;

#define Die(a...) do { fprintf(stderr, "Error: "); fprintf(stderr, a); exit(1);} while(0);

AnswerDb::Entry AnswerDb::Find(int n, int seed) {
    for (int i = 0; i < entries_.size(); i++) {
        const Entry &entry = entries_[i];
        if (entry.n == n && entry.seed == seed)
            return entry;
    }
    return Entry();
}

AnswerDb::AnswerDb() {
    FILE *f = fopen("answers.ssv", "r");
    assert(f);
    char b[1024];
    while (fgets(b, sizeof(b), f)) {
        string s = b;
        assert(s.size() < sizeof(b)-1);
        int p = s.find('#');
        if (p != string::npos)
            s = s.substr(0, p);
        p = 0;
        while (p < s.size() && isspace(s[p]))
            p++;
        s = s.substr(p);
        while (!s.empty() && isspace(s[s.size()-1]))
            s.resize(s.size()-1);
        if (s.empty())
            continue;
        Entry entry;
        assert (3 == sscanf(s.c_str(), "%d %d %f", &entry.n, &entry.seed,
                    &entry.answer));
        if (!Find(entry.n, entry.seed).empty())
            Die("Duplicate entry for n=%d seed=%d\n", entry.n, entry.seed);
        entries_.push_back(entry);
    }
}
bool AnswerDb::CheckAnswer(const string &method, int n, int seed,
        float computed_answer) {
    Entry entry = Find(n, seed);
    if (entry.empty()) {
        printf("Did not find hardcoded answer for method=%s n=%d seed=%d."
                " Calculated: %f\n",
                method.c_str(), n, seed, computed_answer);
        assert(false);
        return false;
    }
    float hardcoded_answer = entry.answer;
    if (fabsf(hardcoded_answer-computed_answer) < 1e-4) {
        printf("Verified answer for method=%s n=%d seed=%d: %f\n",
                method.c_str(), n, seed, computed_answer);
    } else {
        printf("Answer mismatch for method=%s n=%d, seed=%d:"
                " calculated %f vs expected %f\n",
                method.c_str(), n, seed, computed_answer, hardcoded_answer);
        assert(false);
        return false;
    }
    return true;
}
