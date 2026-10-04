package main

import "testing"

func TestParsePeers(t *testing.T) {
	good, err := parsePeers("1=http://a:1, 2=http://b:2 ,3=http://c:3/")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(good) != 3 {
		t.Fatalf("parsed %d peers, want 3", len(good))
	}
	if good[3] != "http://c:3" {
		t.Errorf("trailing slash not trimmed: %q", good[3])
	}

	for _, bad := range []string{
		"", "   ",
		"nope",
		"x=http://a:1",
		"0=http://a:1",
		"1=",
		"1=http://a:1,1=http://b:2",
	} {
		if _, err := parsePeers(bad); err == nil {
			t.Errorf("parsePeers(%q) succeeded, want an error", bad)
		}
	}
}
