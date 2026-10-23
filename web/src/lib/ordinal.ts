/** English ordinal for a whole number: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 92nd. */
export function ordinal(value: number): string {
  const lastTwo = Math.abs(value) % 100
  if (lastTwo >= 11 && lastTwo <= 13) return `${value}th`
  const suffix = { 1: 'st', 2: 'nd', 3: 'rd' }[Math.abs(value) % 10] ?? 'th'
  return `${value}${suffix}`
}
