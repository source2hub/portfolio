// Sample Olympics data for visualizations
export interface OlympicsData {
  year: number;
  country: string;
  gold: number;
  silver: number;
  bronze: number;
  total: number;
  sport: string;
  athlete: string;
  age: number;
  gender: 'M' | 'F';
  height: number;
  weight: number;
  season: 'Summer' | 'Winter';
}

export const medalsByCountry = [
  { country: 'United States', gold: 1180, silver: 959, bronze: 841, total: 2980 },
  { country: 'Soviet Union', gold: 473, silver: 376, bronze: 355, total: 1204 },
  { country: 'Germany', gold: 461, silver: 461, bronze: 454, total: 1376 },
  { country: 'Italy', gold: 259, silver: 229, bronze: 241, total: 729 },
  { country: 'France', gold: 248, silver: 276, bronze: 316, total: 840 },
  { country: 'Great Britain', gold: 236, silver: 272, bronze: 272, total: 780 },
  { country: 'China', gold: 262, silver: 199, bronze: 173, total: 634 },
  { country: 'Australia', gold: 164, silver: 173, bronze: 210, total: 547 },
  { country: 'Hungary', gold: 181, silver: 154, bronze: 176, total: 511 },
  { country: 'Sweden', gold: 148, silver: 170, bronze: 179, total: 497 }
];

export const medalsByYear = [
  { year: 1896, countries: 14, athletes: 241, events: 43, totalMedals: 129 },
  { year: 1900, countries: 24, athletes: 997, events: 95, totalMedals: 285 },
  { year: 1904, countries: 12, athletes: 651, events: 91, totalMedals: 273 },
  { year: 1908, countries: 22, athletes: 2008, events: 110, totalMedals: 330 },
  { year: 1912, countries: 28, athletes: 2407, events: 102, totalMedals: 306 },
  { year: 1920, countries: 29, athletes: 2626, events: 154, totalMedals: 462 },
  { year: 1924, countries: 44, athletes: 3089, events: 126, totalMedals: 378 },
  { year: 1928, countries: 46, athletes: 2883, events: 109, totalMedals: 327 },
  { year: 1932, countries: 37, athletes: 1332, events: 117, totalMedals: 351 },
  { year: 1936, countries: 49, athletes: 3963, events: 129, totalMedals: 387 },
  { year: 1948, countries: 59, athletes: 4104, events: 136, totalMedals: 408 },
  { year: 1952, countries: 69, athletes: 4955, events: 149, totalMedals: 447 },
  { year: 1956, countries: 72, athletes: 3314, events: 151, totalMedals: 453 },
  { year: 1960, countries: 83, athletes: 5338, events: 150, totalMedals: 450 },
  { year: 1964, countries: 93, athletes: 5151, events: 163, totalMedals: 489 },
  { year: 1968, countries: 112, athletes: 5516, events: 172, totalMedals: 516 },
  { year: 1972, countries: 121, athletes: 7134, events: 195, totalMedals: 585 },
  { year: 1976, countries: 92, athletes: 6084, events: 198, totalMedals: 594 },
  { year: 1980, countries: 80, athletes: 5179, events: 203, totalMedals: 609 },
  { year: 1984, countries: 140, athletes: 6829, events: 221, totalMedals: 663 },
  { year: 1988, countries: 159, athletes: 8391, events: 237, totalMedals: 711 },
  { year: 1992, countries: 169, athletes: 9356, events: 257, totalMedals: 771 },
  { year: 1996, countries: 197, athletes: 10318, events: 271, totalMedals: 813 },
  { year: 2000, countries: 199, athletes: 10651, events: 300, totalMedals: 900 },
  { year: 2004, countries: 201, athletes: 10625, events: 301, totalMedals: 903 },
  { year: 2008, countries: 204, athletes: 10942, events: 302, totalMedals: 906 },
  { year: 2012, countries: 204, athletes: 10568, events: 302, totalMedals: 906 },
  { year: 2016, countries: 207, athletes: 11238, events: 306, totalMedals: 918 }
];

export const sportPopularity = [
  { sport: 'Athletics', athletes: 23456, events: 47, countries: 195 },
  { sport: 'Swimming', athletes: 8934, events: 34, countries: 178 },
  { sport: 'Gymnastics', athletes: 5672, events: 18, countries: 134 },
  { sport: 'Cycling', athletes: 4512, events: 22, countries: 156 },
  { sport: 'Wrestling', athletes: 3845, events: 18, countries: 142 },
  { sport: 'Boxing', athletes: 3234, events: 13, countries: 128 },
  { sport: 'Weightlifting', athletes: 2789, events: 15, countries: 98 },
  { sport: 'Rowing', athletes: 2456, events: 14, countries: 87 },
  { sport: 'Football', athletes: 2234, events: 2, countries: 45 },
  { sport: 'Sailing', athletes: 1998, events: 10, countries: 76 }
];

export const genderDistribution = [
  { year: 1896, male: 241, female: 0, total: 241 },
  { year: 1900, male: 975, female: 22, total: 997 },
  { year: 1904, male: 645, female: 6, total: 651 },
  { year: 1908, male: 1971, female: 37, total: 2008 },
  { year: 1912, male: 2359, female: 48, total: 2407 },
  { year: 1920, male: 2561, female: 65, total: 2626 },
  { year: 1924, male: 2954, female: 135, total: 3089 },
  { year: 1928, male: 2606, female: 277, total: 2883 },
  { year: 1932, male: 1206, female: 126, total: 1332 },
  { year: 1936, male: 3632, female: 331, total: 3963 },
  { year: 1948, male: 3714, female: 390, total: 4104 },
  { year: 1952, male: 4436, female: 519, total: 4955 },
  { year: 1956, male: 2938, female: 376, total: 3314 },
  { year: 1960, male: 4727, female: 611, total: 5338 },
  { year: 1964, male: 4473, female: 678, total: 5151 },
  { year: 1968, male: 4735, female: 781, total: 5516 },
  { year: 1972, male: 6075, female: 1059, total: 7134 },
  { year: 1976, male: 4824, female: 1260, total: 6084 },
  { year: 1980, male: 4064, female: 1115, total: 5179 },
  { year: 1984, male: 5263, female: 1566, total: 6829 },
  { year: 1988, male: 6197, female: 2194, total: 8391 },
  { year: 1992, male: 6652, female: 2704, total: 9356 },
  { year: 1996, male: 6806, female: 3512, total: 10318 },
  { year: 2000, male: 6582, female: 4069, total: 10651 },
  { year: 2004, male: 6296, female: 4329, total: 10625 },
  { year: 2008, male: 6305, female: 4637, total: 10942 },
  { year: 2012, male: 5892, female: 4676, total: 10568 },
  { year: 2016, male: 6179, female: 5059, total: 11238 }
];

export const ageDistribution = [
  { ageGroup: '15-20', count: 12456, percentage: 18.2 },
  { ageGroup: '21-25', count: 28934, percentage: 42.3 },
  { ageGroup: '26-30', count: 18672, percentage: 27.3 },
  { ageGroup: '31-35', count: 6234, percentage: 9.1 },
  { ageGroup: '36-40', count: 1567, percentage: 2.3 },
  { ageGroup: '40+', count: 534, percentage: 0.8 }
];

export const countryMedalTrends = [
  { year: 1992, USA: 108, China: 54, Russia: 112, Germany: 82, Australia: 27 },
  { year: 1996, USA: 101, China: 50, Russia: 63, Germany: 65, Australia: 41 },
  { year: 2000, USA: 97, China: 59, Russia: 89, Germany: 56, Australia: 58 },
  { year: 2004, USA: 103, China: 63, Russia: 92, Germany: 48, Australia: 49 },
  { year: 2008, USA: 112, China: 100, Russia: 60, Germany: 41, Australia: 46 },
  { year: 2012, USA: 104, China: 91, Russia: 68, Germany: 44, Australia: 35 },
  { year: 2016, USA: 121, China: 70, Russia: 56, Germany: 42, Australia: 29 }
];

export const medalEfficiency = [
  { country: 'Norway', athletesPerMedal: 2.3, medals: 148, athletes: 340 },
  { country: 'Liechtenstein', athletesPerMedal: 2.8, medals: 10, athletes: 28 },
  { country: 'Slovenia', athletesPerMedal: 3.1, medals: 23, athletes: 71 },
  { country: 'Switzerland', athletesPerMedal: 3.4, medals: 62, athletes: 211 },
  { country: 'Denmark', athletesPerMedal: 3.7, medals: 45, athletes: 167 },
  { country: 'Hungary', athletesPerMedal: 4.2, medals: 511, athletes: 2146 },
  { country: 'Sweden', athletesPerMedal: 4.5, medals: 497, athletes: 2238 },
  { country: 'Bulgaria', athletesPerMedal: 4.8, medals: 221, athletes: 1061 },
  { country: 'Jamaica', athletesPerMedal: 5.1, medals: 78, athletes: 398 },
  { country: 'Cuba', athletesPerMedal: 5.3, medals: 226, athletes: 1198 }
];