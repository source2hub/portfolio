import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  ScatterChart,
  Scatter
} from "recharts";
import {
  medalsByCountry,
  medalsByYear,
  sportPopularity,
  genderDistribution,
  ageDistribution,
  countryMedalTrends,
  medalEfficiency
} from "@/data/olympics-sample-data";

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#f97316', '#06b6d4', '#84cc16'];

export default function OlympicsVisualizations() {
  const [selectedCountries, setSelectedCountries] = useState<string[]>(['USA', 'China', 'Russia', 'Germany']);
  const [selectedTimeframe, setSelectedTimeframe] = useState('all');

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="font-semibold text-foreground">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8" data-testid="olympics-visualizations">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-4">Interactive Olympics Data Analysis</h2>
        <p className="text-muted-foreground max-w-3xl mx-auto">
          Explore comprehensive Olympic Games data through interactive visualizations. 
          Discover trends, patterns, and insights from over a century of Olympic competition.
        </p>
      </div>

      <Tabs defaultValue="medals" className="w-full" data-testid="visualization-tabs">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="medals" data-testid="tab-medals">Medal Distribution</TabsTrigger>
          <TabsTrigger value="trends" data-testid="tab-trends">Historical Trends</TabsTrigger>
          <TabsTrigger value="sports" data-testid="tab-sports">Sports Popularity</TabsTrigger>
          <TabsTrigger value="demographics" data-testid="tab-demographics">Demographics</TabsTrigger>
          <TabsTrigger value="performance" data-testid="tab-performance">Country Performance</TabsTrigger>
          <TabsTrigger value="efficiency" data-testid="tab-efficiency">Medal Efficiency</TabsTrigger>
        </TabsList>

        <TabsContent value="medals" className="space-y-6">
          <Card data-testid="medals-by-country-card">
            <CardHeader>
              <CardTitle>Total Medals by Country (All-Time)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={medalsByCountry.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="country" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="gold" stackId="a" fill="#ffd700" name="Gold" />
                  <Bar dataKey="silver" stackId="a" fill="#c0c0c0" name="Silver" />
                  <Bar dataKey="bronze" stackId="a" fill="#cd7f32" name="Bronze" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card data-testid="medal-breakdown-card">
            <CardHeader>
              <CardTitle>Medal Type Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Gold', value: 33.3, color: '#ffd700' },
                      { name: 'Silver', value: 33.3, color: '#c0c0c0' },
                      { name: 'Bronze', value: 33.4, color: '#cd7f32' }
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {[{ color: '#ffd700' }, { color: '#c0c0c0' }, { color: '#cd7f32' }].map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="space-y-6">
          <Card data-testid="olympics-growth-card">
            <CardHeader>
              <CardTitle>Olympic Games Growth Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={medalsByYear}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="countries" stroke="#3b82f6" name="Countries" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="athletes" stroke="#ef4444" name="Athletes" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="events" stroke="#10b981" name="Events" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card data-testid="gender-participation-card">
            <CardHeader>
              <CardTitle>Gender Participation Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={genderDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Area type="monotone" dataKey="male" stackId="1" stroke="#3b82f6" fill="#3b82f6" name="Male Athletes" />
                  <Area type="monotone" dataKey="female" stackId="1" stroke="#ec4899" fill="#ec4899" name="Female Athletes" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sports" className="space-y-6">
          <Card data-testid="sport-popularity-card">
            <CardHeader>
              <CardTitle>Sport Popularity by Participation</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={sportPopularity} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="sport" type="category" width={100} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="athletes" fill="#3b82f6" name="Total Athletes" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card data-testid="events-by-sport-card">
            <CardHeader>
              <CardTitle>Number of Events by Sport</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={sportPopularity.slice(0, 8)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ sport, events }) => `${sport}: ${events}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="events"
                  >
                    {sportPopularity.slice(0, 8).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="demographics" className="space-y-6">
          <Card data-testid="age-distribution-card">
            <CardHeader>
              <CardTitle>Athlete Age Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={ageDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="ageGroup" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#3b82f6" name="Number of Athletes" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card data-testid="age-percentage-card">
            <CardHeader>
              <CardTitle>Age Distribution Percentage</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={ageDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ ageGroup, percentage }) => `${ageGroup}: ${percentage}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="percentage"
                  >
                    {ageDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <div className="mb-4">
            <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Select timeframe" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Years</SelectItem>
                <SelectItem value="recent">Recent (2000-2016)</SelectItem>
                <SelectItem value="cold-war">Cold War Era (1960-1990)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Card data-testid="country-medal-trends-card">
            <CardHeader>
              <CardTitle>Country Medal Trends (1992-2016)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={countryMedalTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line type="monotone" dataKey="USA" stroke="#3b82f6" strokeWidth={3} name="United States" />
                  <Line type="monotone" dataKey="China" stroke="#ef4444" strokeWidth={3} name="China" />
                  <Line type="monotone" dataKey="Russia" stroke="#10b981" strokeWidth={3} name="Russia" />
                  <Line type="monotone" dataKey="Germany" stroke="#f59e0b" strokeWidth={3} name="Germany" />
                  <Line type="monotone" dataKey="Australia" stroke="#8b5cf6" strokeWidth={3} name="Australia" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="efficiency" className="space-y-6">
          <Card data-testid="medal-efficiency-card">
            <CardHeader>
              <CardTitle>Medal Efficiency: Athletes per Medal</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={medalEfficiency}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="athletes" name="Total Athletes" />
                  <YAxis dataKey="medals" name="Total Medals" />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
                            <p className="font-semibold text-foreground">{data.country}</p>
                            <p className="text-sm text-muted-foreground">Athletes: {data.athletes}</p>
                            <p className="text-sm text-muted-foreground">Medals: {data.medals}</p>
                            <p className="text-sm text-primary font-medium">
                              Efficiency: {data.athletesPerMedal} athletes/medal
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter dataKey="medals" fill="#3b82f6" />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card data-testid="efficiency-ranking-card">
            <CardHeader>
              <CardTitle>Most Efficient Countries (Fewest Athletes per Medal)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={medalEfficiency.slice(0, 10)} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="country" type="category" width={100} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="athletesPerMedal" fill="#10b981" name="Athletes per Medal" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Card className="bg-primary/5" data-testid="data-insights-card">
        <CardHeader>
          <CardTitle>Key Insights from the Data</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-foreground mb-2">🏆 Medal Distribution</h4>
              <p className="text-sm text-muted-foreground">
                The United States leads in total medals with 2,980, followed by the Soviet Union and Germany. 
                This reflects both longevity of participation and athletic excellence.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-2">📈 Olympic Growth</h4>
              <p className="text-sm text-muted-foreground">
                Participation has grown dramatically from 241 athletes in 1896 to over 11,000 in 2016, 
                showing the Olympics' evolution into a truly global event.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-2">👥 Gender Equality</h4>
              <p className="text-sm text-muted-foreground">
                Female participation has grown from 0% in 1896 to nearly 45% in 2016, 
                demonstrating significant progress toward gender equality in sports.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-2">🎯 Efficiency Leaders</h4>
              <p className="text-sm text-muted-foreground">
                Smaller nations like Norway and Liechtenstein show remarkable efficiency with 
                the lowest athletes-per-medal ratios, highlighting focused athletic programs.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}