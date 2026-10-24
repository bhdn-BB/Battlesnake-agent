// JSON-lines, persistent, batched adapter. All game mechanics belong to rules.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	rules "github.com/BattlesnakeOfficial/rules"
	"github.com/BattlesnakeOfficial/rules/maps"
)

const revision = "87e094e2e1c224e9dea67743fd3c2249137c4057"

var directions = []string{"up", "down", "left", "right"}

type Point struct {
	X int `json:"x"`
	Y int `json:"y"`
}
type Snake struct {
	ID     string  `json:"id"`
	Health int     `json:"health"`
	Body   []Point `json:"body"`
	Cause  string  `json:"eliminated_cause"`
	By     string  `json:"eliminated_by"`
	On     int     `json:"eliminated_on_turn"`
}
type Board struct {
	Turn    int     `json:"turn"`
	Width   int     `json:"width"`
	Height  int     `json:"height"`
	Food    []Point `json:"food"`
	Hazards []Point `json:"hazards"`
	Snakes  []Snake `json:"snakes"`
}
type Settings struct {
	MinimumFood     int `json:"minimum_food"`
	FoodSpawnChance int `json:"food_spawn_chance"`
	HazardDamage    int `json:"hazard_damage"`
}
type Snapshot struct {
	Board    Board    `json:"board"`
	Seed     int64    `json:"seed"`
	Solo     bool     `json:"solo"`
	Settings Settings `json:"settings"`
	Revision string   `json:"revision"`
}
type Request struct {
	Op       string         `json:"op"`
	Env      string         `json:"env"`
	Seed     int64          `json:"seed"`
	Snakes   int            `json:"snakes"`
	Settings *Settings      `json:"settings"`
	Actions  map[string]int `json:"actions"`
	Snapshot *Snapshot      `json:"snapshot"`
}
type Response struct {
	Snapshot *Snapshot `json:"snapshot,omitempty"`
	GameOver bool      `json:"game_over"`
	Error    string    `json:"error,omitempty"`
}

func points(ps []Point) []rules.Point {
	out := make([]rules.Point, len(ps))
	for i, p := range ps {
		out[i] = rules.Point{X: p.X, Y: p.Y}
	}
	return out
}
func plain(ps []rules.Point) []Point {
	out := make([]Point, len(ps))
	for i, p := range ps {
		out[i] = Point{p.X, p.Y}
	}
	return out
}
func fromBoard(b *rules.BoardState) Board {
	out := Board{b.Turn, b.Width, b.Height, plain(b.Food), plain(b.Hazards), make([]Snake, len(b.Snakes))}
	for i, s := range b.Snakes {
		out.Snakes[i] = Snake{s.ID, s.Health, plain(s.Body), s.EliminatedCause, s.EliminatedBy, s.EliminatedOnTurn}
	}
	return out
}
func (s Snapshot) board() *rules.BoardState {
	b := rules.NewBoardState(s.Board.Width, s.Board.Height)
	b.Turn = s.Board.Turn
	b.Food = points(s.Board.Food)
	b.Hazards = points(s.Board.Hazards)
	for _, v := range s.Board.Snakes {
		b.Snakes = append(b.Snakes, rules.Snake{ID: v.ID, Health: v.Health, Body: points(v.Body), EliminatedCause: v.Cause, EliminatedBy: v.By, EliminatedOnTurn: v.On})
	}
	return b
}
func (s Snapshot) settings() rules.Settings {
	return rules.NewSettings(map[string]string{
		rules.ParamMinimumFood: strconv.Itoa(s.Settings.MinimumFood), rules.ParamFoodSpawnChance: strconv.Itoa(s.Settings.FoodSpawnChance), rules.ParamHazardDamagePerTurn: strconv.Itoa(s.Settings.HazardDamage),
	}).WithSeed(s.Seed)
}
func over(b *rules.BoardState, solo bool) bool {
	alive := 0
	for _, s := range b.Snakes {
		if s.EliminatedCause == rules.NotEliminated {
			alive++
		}
	}
	if solo {
		return alive == 0
	}
	return alive <= 1
}
func advance(s Snapshot, actions map[string]int) (Snapshot, bool, error) {
	b := s.board()
	if over(b, s.Solo) {
		return s, true, fmt.Errorf("cannot step completed game")
	}
	moves := []rules.SnakeMove{}
	for _, snake := range b.Snakes {
		if snake.EliminatedCause != "" {
			continue
		}
		a, ok := actions[snake.ID]
		if !ok || a < 0 || a > 3 {
			return s, false, fmt.Errorf("missing or invalid action for %s", snake.ID)
		}
		moves = append(moves, rules.SnakeMove{ID: snake.ID, Move: directions[a]})
	}
	settings := s.settings()
	// Standard map PreUpdateBoard is a no-op. Rules execute movement simultaneously.
	r := rules.NewRulesetBuilder().WithSettings(settings).WithSolo(s.Solo).NamedRuleset(rules.GameTypeStandard)
	_, next, err := r.Execute(b, moves)
	if err != nil {
		return s, false, err
	}
	next.Turn = b.Turn + 1
	// Apply the official map's post-turn spawning, even on the final resolved turn.
	editor := maps.NewBoardStateEditor(next)
	if err = (maps.StandardMap{}).PostUpdateBoard(next, settings, editor); err != nil {
		return s, false, err
	}
	s.Board = fromBoard(next)
	return s, over(next, s.Solo), nil
}
func handle(q Request, envs map[string]Snapshot) Response {
	var s Snapshot
	var err error
	var ended bool
	switch q.Op {
	case "reset":
		if q.Snakes != 1 && q.Snakes != 2 && q.Snakes != 4 {
			return Response{Error: "snakes must be 1, 2, or 4"}
		}
		seed := q.Seed
		if seed == 0 {
			seed = 1
		} // engine uses global RNG for zero; canonicalize explicitly.
		settings := Settings{1, 15, 14}
		if q.Settings != nil {
			settings = *q.Settings
		}
		s = Snapshot{Seed: seed, Solo: q.Snakes == 1, Settings: settings, Revision: revision}
		ids := []string{}
		for i := 0; i < q.Snakes; i++ {
			ids = append(ids, fmt.Sprintf("s%d", i))
		}
		b := rules.NewBoardState(11, 11)
		rules.InitializeSnakes(b, ids)
		err = (maps.StandardMap{}).SetupBoard(b, s.settings(), maps.NewBoardStateEditor(b))
		if err == nil {
			s.Board = fromBoard(b)
		}
	case "restore":
		if q.Snapshot == nil || q.Snapshot.Revision != revision {
			return Response{Error: "snapshot revision mismatch"}
		}
		s = *q.Snapshot
	case "snapshot", "step":
		var ok bool
		s, ok = envs[q.Env]
		if !ok {
			return Response{Error: "unknown environment"}
		}
		if q.Op == "step" {
			s, ended, err = advance(s, q.Actions)
		}
	case "successor":
		if q.Snapshot == nil {
			return Response{Error: "successor requires visible snapshot"}
		}
		s = *q.Snapshot
		s.Seed = q.Seed
		if s.Seed == 0 {
			return Response{Error: "counterfactual seed must be nonzero"}
		}
		s, ended, err = advance(s, q.Actions)
	default:
		return Response{Error: "unknown operation"}
	}
	if err != nil {
		return Response{Error: err.Error()}
	}
	if q.Op != "successor" {
		envs[q.Env] = s
	}
	return Response{Snapshot: &s, GameOver: ended}
}
func main() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 65536), 32*1024*1024)
	writer := bufio.NewWriter(os.Stdout)
	envs := map[string]Snapshot{}
	for scanner.Scan() {
		var requests []Request
		if err := json.Unmarshal(scanner.Bytes(), &requests); err != nil {
			json.NewEncoder(writer).Encode([]Response{{Error: err.Error()}})
			writer.Flush()
			continue
		}
		responses := make([]Response, len(requests))
		for i, q := range requests {
			responses[i] = handle(q, envs)
		}
		json.NewEncoder(writer).Encode(responses)
		writer.Flush()
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
