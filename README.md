# Visualization README

  This directory contains the interactive music-sampling visualization and its scrollytelling wrapper.

  ## What This Project Is

  The visualization maps how recent chart hits revive older songs through:

  - samples
  - interpolations
  - covers

  Each connection is drawn as an arc from an older source song on the left to a newer charting revival on the right. The chart is organized so stronger chart
  performance appears higher on the screen.

  There are two main entry points:

  - `index.html`: the core interactive visualization
  - `scrolly.html`: the narrative intro layer that drives the visualization step by step

  ## Main Files

  - `index.html`
    The main D3 visualization. Handles layout, filtering, hover/click behavior, inline labels, timeline rendering, and the programmatic `window.vizAPI` hooks
  used by the scrolly page.

  - `scrolly.html`
    A fixed-card storytelling layer that loads `index.html` in an iframe and controls it via `iframe.contentWindow.vizAPI`.

  - `sample_pairs_clean.csv`
    The source dataset used by the visualization.

  ## How The Visualization Works

  ### Left Side

  The left side shows the older source songs being sampled, interpolated, or covered.

  - Bubble size reflects chart longevity or prominence
  - Bubble color reflects relationship type
  - Songs are grouped by release year
  - Vertical position tracks chart strength buckets

  ### Right Side

  The right side shows the newer revival songs.

  - Each revival node represents one charting song
  - Connected source songs fan into that revival
  - Hover and lock states reveal inline cards and highlight connected arcs

  ### Relationship Colors

  - `samples`: `#e07a5f`
  - `interpolates`: `#3d8b8b`
  - `cover`: `#9b72aa`

  ## Filtering And Focus

  The visualization is year-based. The visible UI exposes yearly filters for the available chart years.

  Even though the UI no longer includes an `ALL` button, the underlying code still supports an `"ALL"` year mode internally.

  Focus behavior exists in two forms:

  - normal interaction: hover and click on nodes/arcs
  - programmatic interaction: `window.vizAPI`

  ## Programmatic API

  `index.html` exposes a global API after the chart renders:

  ```js
  window.vizAPI = {
    setYear,
    focusSong,
    focusConnection,
    clearFocus,
    setScrollyMode
  }

  ### setYear(year)

  Applies the same logic as clicking a year button.

  ### focusSong(revivalSongName, revivalArtistName)

  Focuses a revival song and shows its connected sources.

  ### focusConnection(revivalSongName, originalSongName)

  Focuses one specific connection between a revival and an original source song.

  ### clearFocus()

  Removes lock/focus state and restores the default view.

  ### setScrollyMode(active)

  Used by the scrollytelling layer to switch the chart into a more isolated, narrative-friendly focus mode.

  ## Scrollytelling Layer

  scrolly.html is a self-contained storytelling wrapper around the chart.

  It:

  - embeds index.html in an iframe
  - advances through a fixed sequence of story cards
  - calls vizAPI methods to move between years and highlight specific songs or song-to-song connections
  - blocks direct interaction with the chart until the user exits the intro

  The final card hands the user off to the live chart so they can explore freely.

  ## Interaction Notes

  - Hovering a node or arc reveals inline song cards
  - Clicking locks the current focus
  - Clicking the background clears the lock
  - Some original/source nodes can be split-color nodes when the same source song is used in multiple relationship types
  - Collision handling is built in for overlapping source nodes

  ## Data Notes

  The dataset is a curated list of revival/source relationships. Important fields include:

  - revival_song
  - revival_artist
  - revival_peak
  - revival_chart_year
  - original_song
  - original_artist
  - original_release_year
  - relationship_type
  - years_between

  The chart generally renders Top 50 revival songs for the active year, with a small forced-focus exception used in certain scrollytelling steps when a
  specific connection needs to be shown even if it falls outside the default cutoff.

  ## Running Locally

  Because this is a static HTML/D3 project, the simplest local workflow is to serve the viz/ directory with a local web server.

  Example:


  Then open:

  - http://localhost:8000/index.html
  - http://localhost:8000/scrolly.html

  ## Deployment
  For example, a Netlify deploy would typically use:

  netlify deploy --prod --dir viz

  from the revival-tracks/ project root.

  ## Editing Notes

  If you modify the chart:

  - keep index.html focused on visualization logic
  - keep scrolly.html focused on narrative orchestration
  - preserve the vizAPI contract if the scrolly page depends on it
  - be careful with any async render/focus timing, because the chart uses D3 transitions and those can override state if not reapplied deliberately

  If you modify copy:

  - update the steps array in scrolly.html
  - verify any focusConnection() step still matches the exact song titles in sample_pairs_clean.csv

  ## Summary

  This project is a narrative D3 visualization about how modern hits revive older music. index.html is the chart. scrolly.html is the guided story. The CSV
  drives both.
