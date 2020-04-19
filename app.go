// Package p contains an HTTP Cloud Function.
package p

import (
  "cloud.google.com/go/datastore"
  "context"
  "encoding/json"
  "fmt"
  "io/ioutil"
  "net/http"
  "strings"
)

const Project = "dota-draft"
const PublicMatchesURL = "https://api.opendota.com/api/publicMatches"

type Match struct {
  // General
  MatchId     int  `json:"match_id"`
  MatchSeqNum int  `json:"match_seq_num"`
  RadiantWin  bool `json:"radiant_win"`
  StartTime   int  `datastore:",noindex" json:"start_time"`
  Duration    int  `datastore:",noindex" json:"duration"`

  // Rank metadata
  AvgMMR      int `datastore:",noindex" json:"avg_mmr"`
  NumMMR      int `datastore:",noindex" json:"num_mmr"`
  AvgRankTier int `datastore:",noindex" json:"avg_rank_tier"`
  NumRankTier int `datastore:",noindex" json:"num_rank_tier"`
  Cluster     int `datastore:",noindex" json:"cluster"`

  // Game mode
  LobbyType int `json:"lobby_type"`
  GameMode  int `json:"game_mode"`

  // Team hero compositions
  RadiantTeam []string
  DireTeam    []string
}

func (m *Match) UnmarshalJSON(data []byte) error {
  type Alias Match
  aux := &struct {
    RadiantTeam string `json:"radiant_team"`
    DireTeam    string `json:"dire_team"`
    *Alias
  }{
    Alias: (*Alias)(m),
  }
  if err := json.Unmarshal(data, &aux); err != nil {
    return err
  }
  m.RadiantTeam = strings.Split(aux.RadiantTeam, ",")
  m.DireTeam = strings.Split(aux.DireTeam, ",")
  return nil
}

var ctx context.Context
var client datastore.Client

// Init runs during package initialization.
func init() {
  var err error
  ctx = context.Background()
  client, err = datastore.NewClient(ctx, Project)
  if err != nil {
    panic(fmt.Sprintf("Error initializing datastore client: %v", err))
  }
}

func PublicMatches(w http.ResponseWriter, r *http.Request) {
  resp, err := http.Get(PublicMatchesURL)

  if err != nil {
    panic(err.Error())
  }

  body, err := ioutil.ReadAll(resp.Body)
  if err != nil {
    panic(err.Error())
  }

  var matches []Match
  json.Unmarshal(body, &matches)

  for _, match := range matches {
    key := Key{
      Kind: "match",
      ID:   match.MatchId,
    }

    if _, err := client.Put(ctx, key, match); err != nil {
      panic(err.Error())
    }
  }

  output := []byte(fmt.Sprintf(`{"num_matches": %d}`, len(matches)))
  w.Header().Set("Content-Type", "application/json")
  w.Write(output)
}
