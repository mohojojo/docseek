"""Generated discovery programs: an agent explores a site once and writes a plain-Python program that finds the
documents a goal asks for; later crawls replay the program with no model.

  fetcher   the only route to the web, for the agent and its programs
  sandbox   runs a program in a child process
  explorer  the agent that writes a program
  programs  the program store, hybrid discovery (program first, crawl as the safety net) and the drift check
"""
