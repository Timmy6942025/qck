package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// PkgManager defines a package manager and its search behavior
type PkgManager struct {
	Name       string
	DetectCmd  string
	DetectArgs []string
	SearchCmd  string
	SearchArgs func(query string) []string
	ParseLine  func(line string) string
}

var managers = []PkgManager{
	// Linux
	{
		Name: "apt", DetectCmd: "apt", DetectArgs: []string{"--version"},
		SearchCmd:  "apt-cache",
		SearchArgs: func(q string) []string { return []string{"search", "--names-only", q} },
		ParseLine: func(l string) string {
			fields := strings.Fields(l)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "dnf", DetectCmd: "dnf", DetectArgs: []string{"--version"},
		SearchCmd:  "dnf",
		SearchArgs: func(q string) []string { return []string{"search", "-q", q} },
		ParseLine: func(l string) string {
			if i := strings.Index(l, "."); i > 0 {
				return l[:i]
			}
			fields := strings.Fields(l)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "pacman", DetectCmd: "pacman", DetectArgs: []string{"--version"},
		SearchCmd:  "pacman",
		SearchArgs: func(q string) []string { return []string{"-Ss", q} },
		ParseLine: func(l string) string {
			if strings.HasPrefix(l, " ") || strings.HasPrefix(l, "\t") {
				return "" // description line
			}
			parts := strings.SplitN(l, "/", 2)
			if len(parts) == 2 {
				name := strings.SplitN(parts[1], " ", 2)[0]
				return strings.TrimSpace(name)
			}
			return ""
		},
	},
	{
		Name: "apk", DetectCmd: "apk", DetectArgs: []string{"--version"},
		SearchCmd:  "apk",
		SearchArgs: func(q string) []string { return []string{"search", q} },
		ParseLine: func(l string) string {
			return strings.TrimSpace(l)
		},
	},
	{
		Name: "zypper", DetectCmd: "zypper", DetectArgs: []string{"--version"},
		SearchCmd:  "zypper",
		SearchArgs: func(q string) []string { return []string{"search", q} },
		ParseLine: func(l string) string {
			// Skip header lines
			if strings.HasPrefix(l, "S") || strings.Contains(l, "---") {
				return ""
			}
			parts := strings.Split(l, "|")
			if len(parts) >= 2 {
				return strings.TrimSpace(parts[1])
			}
			return ""
		},
	},
	// macOS
	{
		Name: "brew", DetectCmd: "brew", DetectArgs: []string{"--version"},
		SearchCmd:  "brew",
		SearchArgs: func(q string) []string { return []string{"search", q} },
		ParseLine: func(l string) string {
			return strings.TrimSpace(l)
		},
	},
	// Windows
	{
		Name: "winget", DetectCmd: "winget", DetectArgs: []string{"--version"},
		SearchCmd:  "winget",
		SearchArgs: func(q string) []string { return []string{"search", "--name", q, "--disable-interactivity"} },
		ParseLine: func(l string) string {
			// Skip header and separator lines
			if strings.HasPrefix(l, "Name") || strings.HasPrefix(l, "-") {
				return ""
			}
			fields := strings.Fields(l)
			if len(fields) >= 2 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "scoop", DetectCmd: "scoop", DetectArgs: []string{"--version"},
		SearchCmd:  "scoop",
		SearchArgs: func(q string) []string { return []string{"search", q} },
		ParseLine: func(l string) string {
			l = strings.TrimSpace(l)
			// Skip header lines
			if strings.HasPrefix(l, "Results") || strings.HasPrefix(l, "-") {
				return ""
			}
			fields := strings.Fields(l)
			if len(fields) >= 1 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "choco", DetectCmd: "choco", DetectArgs: []string{"--version"},
		SearchCmd:  "choco",
		SearchArgs: func(q string) []string { return []string{"search", q, "--limit-output"} },
		ParseLine: func(l string) string {
			parts := strings.SplitN(l, "|", 2)
			if len(parts) >= 1 && parts[0] != "" {
				return parts[0]
			}
			return ""
		},
	},
	// Cross-platform / Language
	{
		Name: "npm", DetectCmd: "npm", DetectArgs: []string{"--version"},
		SearchCmd:  "npm",
		SearchArgs: func(q string) []string { return []string{"search", "--parseable", q} },
		ParseLine: func(l string) string {
			parts := strings.Split(l, "\t")
			if len(parts) >= 1 && parts[0] != "" {
				return parts[0]
			}
			return ""
		},
	},
	{
		Name: "pip", DetectCmd: "pip", DetectArgs: []string{"--version"},
		SearchCmd:  "pip",
		SearchArgs: func(q string) []string { return []string{"index", "versions", q} },
		ParseLine: func(l string) string {
			fields := strings.Fields(l)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "cargo", DetectCmd: "cargo", DetectArgs: []string{"--version"},
		SearchCmd:  "cargo",
		SearchArgs: func(q string) []string { return []string{"search", q, "--limit", "20"} },
		ParseLine: func(l string) string {
			fields := strings.SplitN(l, " ", 2)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "gem", DetectCmd: "gem", DetectArgs: []string{"--version"},
		SearchCmd:  "gem",
		SearchArgs: func(q string) []string { return []string{"search", q, "--remote"} },
		ParseLine: func(l string) string {
			fields := strings.Fields(l)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
	{
		Name: "go", DetectCmd: "go", DetectArgs: []string{"version"},
		SearchCmd:  "go",
		SearchArgs: func(q string) []string { return []string{"list", "-m", "-versions", q} },
		ParseLine: func(l string) string {
			fields := strings.SplitN(l, " ", 2)
			if len(fields) > 0 {
				return fields[0]
			}
			return ""
		},
	},
}

// ScoredResult represents a package search result with fuzzy match score
type ScoredResult struct {
	Name    string
	Manager string
	Score   int
}

// fuzzyScore returns a score > 0 if pattern fuzzy-matches target
func fuzzyScore(pattern, target string) int {
	if pattern == "" {
		return 1
	}

	pLower := strings.ToLower(pattern)
	tLower := strings.ToLower(target)
	pRunes := []rune(pLower)
	tRunes := []rune(tLower)
	tOriginal := []rune(target)

	if len(pRunes) > len(tRunes) {
		return 0
	}

	pi := 0
	score := 0
	prevMatch := false

	for ti := 0; ti < len(tRunes) && pi < len(pRunes); ti++ {
		if tRunes[ti] == pRunes[pi] {
			score += 1

			if prevMatch {
				score += 3
			}

			// Start of word bonus
			if ti == 0 || tRunes[ti-1] == '-' || tRunes[ti-1] == '_' || tRunes[ti-1] == '.' ||
				(unicode.IsLower(tOriginal[ti-1]) && unicode.IsUpper(tOriginal[ti])) {
				score += 5
			}

			// Exact case bonus
			if len(tOriginal) > ti && len([]rune(pattern)) > pi && tOriginal[ti] == []rune(pattern)[pi] {
				score += 1
			}

			pi++
			prevMatch = true
		} else {
			prevMatch = false
		}
	}

	if pi < len(pRunes) {
		return 0
	}

	// Prefer shorter names
	score += 10 - min(10, len(tRunes)-len(pRunes))

	return score
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// detectManagers returns available package managers on the system
func detectManagers() []PkgManager {
	var found []PkgManager
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, m := range managers {
		wg.Add(1)
		go func(m PkgManager) {
			defer wg.Done()
			cmd := exec.Command(m.DetectCmd, m.DetectArgs...)
			cmd.Stdout = nil
			cmd.Stderr = nil
			if err := cmd.Run(); err == nil {
				mu.Lock()
				found = append(found, m)
				mu.Unlock()
			}
		}(m)
	}

	wg.Wait()
	return found
}

// searchManager searches for packages using a specific package manager
func searchManager(m PkgManager, query string) []string {
	var results []string

	cmd := exec.Command(m.SearchCmd, m.SearchArgs(query)...)
	cmd.Stdout = nil
	cmd.Stderr = nil

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return results
	}

	if err := cmd.Start(); err != nil {
		return results
	}

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		if name := m.ParseLine(line); name != "" {
			results = append(results, name)
		}
	}

	cmd.Wait()
	return results
}

// searchAllManagers searches across all available package managers
func searchAllManagers(managers []PkgManager, query string, maxResults int) []ScoredResult {
	var wg sync.WaitGroup
	resultsChan := make(chan struct {
		manager string
		names   []string
	}, len(managers))

	for _, m := range managers {
		wg.Add(1)
		go func(m PkgManager) {
			defer wg.Done()
			names := searchManager(m, query)
			resultsChan <- struct {
				manager string
				names   []string
			}{m.Name, names}
		}(m)
	}

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	var allResults []ScoredResult
	for result := range resultsChan {
		for _, name := range result.names {
			score := fuzzyScore(query, name)
			if score > 0 {
				allResults = append(allResults, ScoredResult{
					Name:    name,
					Manager: result.manager,
					Score:   score,
				})
			}
		}
	}

	// Sort by score descending
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	if len(allResults) > maxResults {
		allResults = allResults[:maxResults]
	}

	return allResults
}

func main() {
	// Define flags
	query := flag.String("q", "", "Search query (required)")
	manager := flag.String("m", "", "Specific package manager to use (optional)")
	limit := flag.Int("n", 20, "Maximum number of results")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Parse()

	if *query == "" {
		fmt.Println("Usage: pkgsearch -q <query> [-m <manager>] [-n <limit>] [-v]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nAvailable package managers will be auto-detected.")
		os.Exit(1)
	}

	// Detect available managers
	available := detectManagers()

	if *verbose {
		fmt.Printf("Detected package managers: ")
		for i, m := range available {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Print(m.Name)
		}
		fmt.Println()
	}

	if len(available) == 0 {
		fmt.Println("No package managers detected on this system.")
		fmt.Printf("Running on: %s\n", runtime.GOOS)
		os.Exit(1)
	}

	// Filter by manager if specified
	var searchManagers []PkgManager
	if *manager != "" {
		found := false
		for _, m := range available {
			if strings.ToLower(m.Name) == strings.ToLower(*manager) {
				searchManagers = []PkgManager{m}
				found = true
				break
			}
		}
		if !found {
			fmt.Printf("Package manager '%s' not found or not available.\n", *manager)
			os.Exit(1)
		}
	} else {
		searchManagers = available
	}

	// Search
	results := searchAllManagers(searchManagers, *query, *limit)

	if len(results) == 0 {
		fmt.Printf("No packages found matching '%s'\n", *query)
		os.Exit(0)
	}

	// Print results
	fmt.Printf("Found %d packages matching '%s':\n\n", len(results), *query)
	for _, r := range results {
		fmt.Printf("[%s] %s (score: %d)\n", r.Manager, r.Name, r.Score)
	}
}
