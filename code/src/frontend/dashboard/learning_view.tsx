/**
 * Learning View Dashboard Component
 * 
 * This component provides an interactive educational interface with trading lessons,
 * progress tracking, quizzes, and learning paths.
 */

'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { 
  BookOpen, 
  Trophy, 
  Target,
  Clock,
  CheckCircle,
  Circle,
  PlayCircle,
  Award,
  TrendingUp,
  Brain,
  BarChart3,
  Lightbulb,
  Star,
  Lock,
  Unlock
} from 'lucide-react'

interface Lesson {
  id: string
  title: string
  description: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  duration: number
  category: string
  completed: boolean
  progress: number
  locked: boolean
  prerequisites: string[]
  topics: string[]
}

interface Quiz {
  id: string
  lessonId: string
  title: string
  questions: number
  completed: boolean
  score: number
  bestScore: number
  attempts: number
}

interface Achievement {
  id: string
  title: string
  description: string
  icon: string
  unlocked: boolean
  unlockedAt?: string
  progress: number
  totalRequired: number
}

interface LearningPath {
  id: string
  title: string
  description: string
  estimatedTime: number
  difficulty: string
  progress: number
  lessons: string[]
  completedLessons: string[]
}

export default function LearningView() {
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null)
  const [selectedPath, setSelectedPath] = useState<string>('foundations')
  const [userProgress, setUserProgress] = useState({
    totalLessons: 24,
    completedLessons: 8,
    totalQuizzes: 18,
    completedQuizzes: 5,
    studyTime: 1250, // minutes
    streak: 7 // days
  })

  const [lessons, setLessons] = useState<Lesson[]>([
    {
      id: 'market-basics',
      title: 'Market Fundamentals',
      description: 'Understanding market structure, participants, and basic trading concepts',
      difficulty: 'beginner',
      duration: 25,
      category: 'Trading Basics',
      completed: true,
      progress: 100,
      locked: false,
      prerequisites: [],
      topics: ['Market Structure', 'Order Types', 'Market Participants', 'Liquidity']
    },
    {
      id: 'technical-analysis',
      title: 'Technical Analysis',
      description: 'Chart patterns, indicators, and technical trading strategies',
      difficulty: 'intermediate',
      duration: 45,
      category: 'Trading Basics',
      completed: true,
      progress: 100,
      locked: false,
      prerequisites: ['market-basics'],
      topics: ['Chart Patterns', 'Technical Indicators', 'Support & Resistance', 'Trend Analysis']
    },
    {
      id: 'risk-management',
      title: 'Risk Management',
      description: 'Position sizing, stop losses, and portfolio risk management',
      difficulty: 'beginner',
      duration: 35,
      category: 'Trading Basics',
      completed: false,
      progress: 60,
      locked: false,
      prerequisites: ['market-basics'],
      topics: ['Position Sizing', 'Stop Losses', 'Risk/Reward Ratio', 'Portfolio Management']
    },
    {
      id: 'options-fundamentals',
      title: 'Options Fundamentals',
      description: 'Understanding options, Greeks, and basic options strategies',
      difficulty: 'intermediate',
      duration: 40,
      category: 'Advanced Trading',
      completed: false,
      progress: 30,
      locked: false,
      prerequisites: ['technical-analysis', 'risk-management'],
      topics: ['Options Basics', 'The Greeks', 'Options Pricing', 'Basic Strategies']
    },
    {
      id: 'machine-learning-basics',
      title: 'Machine Learning for Trading',
      description: 'Introduction to ML concepts and applications in trading',
      difficulty: 'advanced',
      duration: 50,
      category: 'AI & Algorithms',
      completed: false,
      progress: 0,
      locked: true,
      prerequisites: ['technical-analysis', 'risk-management'],
      topics: ['Supervised Learning', 'Feature Engineering', 'Model Evaluation', 'Trading Applications']
    },
    {
      id: 'reinforcement-learning',
      title: 'Reinforcement Learning',
      description: 'Deep dive into RL for autonomous trading agents',
      difficulty: 'advanced',
      duration: 60,
      category: 'AI & Algorithms',
      completed: false,
      progress: 0,
      locked: true,
      prerequisites: ['machine-learning-basics'],
      topics: ['RL Fundamentals', 'Trading Environments', 'Policy Learning', 'Advanced Strategies']
    }
  ])

  const [quizzes, setQuizzes] = useState<Quiz[]>([
    {
      id: 'market-basics-quiz',
      lessonId: 'market-basics',
      title: 'Market Fundamentals Quiz',
      questions: 10,
      completed: true,
      score: 85,
      bestScore: 92,
      attempts: 2
    },
    {
      id: 'technical-analysis-quiz',
      lessonId: 'technical-analysis',
      title: 'Technical Analysis Quiz',
      questions: 15,
      completed: true,
      score: 78,
      bestScore: 88,
      attempts: 3
    },
    {
      id: 'risk-management-quiz',
      lessonId: 'risk-management',
      title: 'Risk Management Quiz',
      questions: 12,
      completed: false,
      score: 0,
      bestScore: 0,
      attempts: 0
    }
  ])

  const [achievements, setAchievements] = useState<Achievement[]>([
    {
      id: 'first-lesson',
      title: 'First Steps',
      description: 'Complete your first lesson',
      icon: '🎯',
      unlocked: true,
      unlockedAt: '2024-01-15',
      progress: 1,
      totalRequired: 1
    },
    {
      id: 'week-streak',
      title: 'Week Warrior',
      description: 'Maintain a 7-day learning streak',
      icon: '🔥',
      unlocked: true,
      unlockedAt: '2024-01-22',
      progress: 7,
      totalRequired: 7
    },
    {
      id: 'quiz-master',
      title: 'Quiz Master',
      description: 'Score 90% or higher on 5 quizzes',
      icon: '🧠',
      unlocked: false,
      progress: 2,
      totalRequired: 5
    },
    {
      id: 'options-expert',
      title: 'Options Expert',
      description: 'Complete all options trading lessons',
      icon: '📊',
      unlocked: false,
      progress: 0,
      totalRequired: 3
    },
    {
      id: 'ai-trader',
      title: 'AI Trader',
      description: 'Complete all machine learning lessons',
      icon: '🤖',
      unlocked: false,
      progress: 0,
      totalRequired: 4
    }
  ])

  const [learningPaths, setLearningPaths] = useState<LearningPath[]>([
    {
      id: 'foundations',
      title: 'Trading Foundations',
      description: 'Master the fundamentals of trading and market analysis',
      estimatedTime: 180,
      difficulty: 'Beginner',
      progress: 65,
      lessons: ['market-basics', 'technical-analysis', 'risk-management'],
      completedLessons: ['market-basics', 'technical-analysis']
    },
    {
      id: 'advanced-trading',
      title: 'Advanced Trading Strategies',
      description: 'Learn sophisticated trading techniques and options strategies',
      estimatedTime: 240,
      difficulty: 'Intermediate',
      progress: 25,
      lessons: ['options-fundamentals', 'advanced-technical', 'portfolio-management'],
      completedLessons: []
    },
    {
      id: 'ai-trading',
      title: 'AI & Algorithmic Trading',
      description: 'Master machine learning and AI for automated trading',
      estimatedTime: 300,
      difficulty: 'Advanced',
      progress: 0,
      lessons: ['machine-learning-basics', 'reinforcement-learning', 'neural-networks', 'deep-learning'],
      completedLessons: []
    }
  ])

  const [selectedTab, setSelectedTab] = useState('overview')

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800'
      case 'intermediate': return 'bg-yellow-100 text-yellow-800'
      case 'advanced': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`
  }

  const startLesson = (lessonId: string) => {
    const lesson = lessons.find(l => l.id === lessonId)
    if (lesson && !lesson.locked) {
      setSelectedLesson(lesson)
      // In a real app, this would navigate to the lesson content
    }
  }

  const calculateOverallProgress = () => {
    const completedLessons = lessons.filter(l => l.completed).length
    return (completedLessons / lessons.length) * 100
  }

  const getRecommendedLessons = () => {
    return lessons
      .filter(lesson => !lesson.completed && !lesson.locked)
      .sort((a, b) => b.progress - a.progress)
      .slice(0, 3)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Learning Center</h1>
          <p className="text-muted-foreground">Master trading with interactive lessons and AI-powered insights</p>
        </div>
        <div className="flex items-center space-x-4">
          <Badge variant="outline" className="flex items-center space-x-2">
            <Trophy className="w-4 h-4" />
            <span>Level 3</span>
          </Badge>
          <Badge variant="outline" className="flex items-center space-x-2">
            <Target className="w-4 h-4" />
            <span>{userProgress.streak} day streak</span>
          </Badge>
        </div>
      </div>

      {/* Progress Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Progress</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{calculateOverallProgress().toFixed(0)}%</div>
            <p className="text-xs text-muted-foreground">
              {userProgress.completedLessons} of {userProgress.totalLessons} lessons
            </p>
            <Progress value={calculateOverallProgress()} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Study Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatDuration(userProgress.studyTime)}</div>
            <p className="text-xs text-muted-foreground">
              This month: {formatDuration(450)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Quizzes Passed</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{userProgress.completedQuizzes}</div>
            <p className="text-xs text-muted-foreground">
              Average score: 82.5%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Achievements</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {achievements.filter(a => a.unlocked).length}/{achievements.length}
            </div>
            <p className="text-xs text-muted-foreground">
              2 new this week
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="lessons">Lessons</TabsTrigger>
          <TabsTrigger value="paths">Learning Paths</TabsTrigger>
          <TabsTrigger value="quizzes">Quizzes</TabsTrigger>
          <TabsTrigger value="achievements">Achievements</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Recommended Lessons */}
          <Card>
            <CardHeader>
              <CardTitle>Recommended Lessons</CardTitle>
              <CardDescription>Continue your learning journey with these personalized recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {getRecommendedLessons().map((lesson) => (
                  <Card key={lesson.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <Badge className={getDifficultyColor(lesson.difficulty)}>
                          {lesson.difficulty}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {formatDuration(lesson.duration)}
                        </span>
                      </div>
                      <CardTitle className="text-lg">{lesson.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-3">{lesson.description}</p>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{lesson.progress}%</span>
                        </div>
                        <Progress value={lesson.progress} />
                      </div>
                      <Button 
                        className="w-full mt-3" 
                        onClick={() => startLesson(lesson.id)}
                      >
                        {lesson.progress > 0 ? 'Continue' : 'Start'} Lesson
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Achievements */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Achievements</CardTitle>
              <CardDescription>Your latest accomplishments and milestones</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {achievements.filter(a => a.unlocked).slice(0, 3).map((achievement) => (
                  <div key={achievement.id} className="flex items-center space-x-3 p-3 border rounded-lg">
                    <div className="text-2xl">{achievement.icon}</div>
                    <div>
                      <p className="font-medium">{achievement.title}</p>
                      <p className="text-sm text-muted-foreground">{achievement.description}</p>
                      {achievement.unlockedAt && (
                        <p className="text-xs text-muted-foreground">
                          Unlocked {new Date(achievement.unlockedAt).toLocaleDateString()}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Learning Streak */}
          <Card>
            <CardHeader>
              <CardTitle>Learning Streak</CardTitle>
              <CardDescription>Keep up the great work!</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="text-3xl">🔥</div>
                  <div>
                    <p className="text-2xl font-bold">{userProgress.streak} days</p>
                    <p className="text-sm text-muted-foreground">Current streak</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-muted-foreground">Best streak</p>
                  <p className="text-lg font-semibold">14 days</p>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex space-x-1">
                  {Array.from({ length: 7 }, (_, i) => (
                    <div
                      key={i}
                      className={`flex-1 h-8 rounded ${
                        i < userProgress.streak ? 'bg-green-500' : 'bg-gray-200'
                      }`}
                    />
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-2">Last 7 days</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="lessons" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {lessons.map((lesson) => (
              <Card key={lesson.id} className={`${lesson.locked ? 'opacity-75' : ''}`}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <Badge className={getDifficultyColor(lesson.difficulty)}>
                      {lesson.difficulty}
                    </Badge>
                    <div className="flex items-center space-x-2">
                      {lesson.locked ? (
                        <Lock className="w-4 h-4 text-muted-foreground" />
                      ) : (
                        <Unlock className="w-4 h-4 text-green-600" />
                      )}
                      <span className="text-sm text-muted-foreground">
                        {formatDuration(lesson.duration)}
                      </span>
                    </div>
                  </div>
                  <CardTitle className="text-lg">{lesson.title}</CardTitle>
                  <CardDescription>{lesson.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {/* Topics */}
                    <div>
                      <p className="text-sm font-medium mb-2">Topics:</p>
                      <div className="flex flex-wrap gap-1">
                        {lesson.topics.map((topic, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {topic}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Prerequisites */}
                    {lesson.prerequisites.length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">Prerequisites:</p>
                        <div className="flex flex-wrap gap-1">
                          {lesson.prerequisites.map((prereq, index) => {
                            const prereqLesson = lessons.find(l => l.id === prereq)
                            const isCompleted = prereqLesson?.completed || false
                            return (
                              <Badge 
                                key={index} 
                                variant={isCompleted ? "default" : "destructive"}
                                className="text-xs"
                              >
                                {prereqLesson?.title || prereq}
                              </Badge>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Progress */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Progress</span>
                        <span>{lesson.progress}%</span>
                      </div>
                      <Progress value={lesson.progress} />
                    </div>

                    {/* Action Button */}
                    <Button 
                      className="w-full" 
                      disabled={lesson.locked}
                      onClick={() => startLesson(lesson.id)}
                    >
                      {lesson.locked ? (
                        <>
                          <Lock className="w-4 h-4 mr-2" />
                          Locked
                        </>
                      ) : lesson.completed ? (
                        <>
                          <CheckCircle className="w-4 h-4 mr-2" />
                          Review
                        </>
                      ) : lesson.progress > 0 ? (
                        <>
                          <PlayCircle className="w-4 h-4 mr-2" />
                          Continue ({lesson.progress}%)
                        </>
                      ) : (
                        <>
                          <PlayCircle className="w-4 h-4 mr-2" />
                          Start Lesson
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="paths" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {learningPaths.map((path) => (
              <Card key={path.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <Badge variant="outline">{path.difficulty}</Badge>
                    <span className="text-sm text-muted-foreground">
                      {formatDuration(path.estimatedTime)}
                    </span>
                  </div>
                  <CardTitle className="text-xl">{path.title}</CardTitle>
                  <CardDescription>{path.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Progress */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Path Progress</span>
                        <span>{path.progress}%</span>
                      </div>
                      <Progress value={path.progress} />
                      <p className="text-xs text-muted-foreground">
                        {path.completedLessons.length} of {path.lessons.length} lessons completed
                      </p>
                    </div>

                    {/* Lessons in Path */}
                    <div>
                      <p className="text-sm font-medium mb-2">Lessons in this path:</p>
                      <div className="space-y-2">
                        {path.lessons.map((lessonId, index) => {
                          const lesson = lessons.find(l => l.id === lessonId)
                          if (!lesson) return null
                          
                          return (
                            <div key={lessonId} className="flex items-center space-x-2">
                              {lesson.completed ? (
                                <CheckCircle className="w-4 h-4 text-green-600" />
                              ) : (
                                <Circle className="w-4 h-4 text-muted-foreground" />
                              )}
                              <span className="text-sm">{lesson.title}</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>

                    {/* Action Button */}
                    <Button 
                      className="w-full"
                      onClick={() => setSelectedPath(path.id)}
                    >
                      {path.progress === 100 ? (
                        <>
                          <Trophy className="w-4 h-4 mr-2" />
                          Path Completed
                        </>
                      ) : path.progress > 0 ? (
                        <>
                          <PlayCircle className="w-4 h-4 mr-2" />
                          Continue Path
                        </>
                      ) : (
                        <>
                          <PlayCircle className="w-4 h-4 mr-2" />
                          Start Path
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="quizzes" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {quizzes.map((quiz) => {
              const lesson = lessons.find(l => l.id === quiz.lessonId)
              return (
                <Card key={quiz.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <Badge variant="outline">
                        {quiz.questions} questions
                      </Badge>
                      {quiz.completed && (
                        <Badge variant="default">
                          {quiz.score}%
                        </Badge>
                      )}
                    </div>
                    <CardTitle className="text-lg">{quiz.title}</CardTitle>
                    <CardDescription>
                      {lesson?.title || 'Unknown Lesson'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {/* Quiz Stats */}
                      {quiz.completed && (
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">Last Score</p>
                            <p className="font-medium">{quiz.score}%</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Best Score</p>
                            <p className="font-medium text-green-600">{quiz.bestScore}%</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Attempts</p>
                            <p className="font-medium">{quiz.attempts}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Status</p>
                            <p className="font-medium text-green-600">Passed</p>
                          </div>
                        </div>
                      )}

                      {/* Action Button */}
                      <Button 
                        className="w-full"
                        variant={quiz.completed ? "outline" : "default"}
                      >
                        {quiz.completed ? (
                          <>
                            <CheckCircle className="w-4 h-4 mr-2" />
                            Retake Quiz
                          </>
                        ) : (
                          <>
                            <PlayCircle className="w-4 h-4 mr-2" />
                            Start Quiz
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        <TabsContent value="achievements" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {achievements.map((achievement) => (
              <Card 
                key={achievement.id} 
                className={`${achievement.unlocked ? '' : 'opacity-75'} ${
                  !achievement.unlocked && achievement.progress > 0 ? 'border-yellow-200' : ''
                }`}
              >
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="text-3xl">{achievement.icon}</div>
                    {achievement.unlocked ? (
                      <Badge variant="default" className="flex items-center space-x-1">
                        <Trophy className="w-3 h-3" />
                        <span>Unlocked</span>
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="flex items-center space-x-1">
                        <Lock className="w-3 h-3" />
                        <span>Locked</span>
                      </Badge>
                    )}
                  </div>
                  <CardTitle className="text-lg">{achievement.title}</CardTitle>
                  <CardDescription>{achievement.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {/* Progress */}
                    {!achievement.unlocked && achievement.totalRequired > 1 && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{achievement.progress}/{achievement.totalRequired}</span>
                        </div>
                        <Progress 
                          value={(achievement.progress / achievement.totalRequired) * 100} 
                        />
                      </div>
                    )}

                    {/* Unlock Date */}
                    {achievement.unlocked && achievement.unlockedAt && (
                      <div className="text-sm text-muted-foreground">
                        <p>Unlocked on {new Date(achievement.unlockedAt).toLocaleDateString()}</p>
                      </div>
                    )}

                    {/* Status */}
                    <div className="text-center">
                      {achievement.unlocked ? (
                        <div className="flex items-center justify-center space-x-2 text-green-600">
                          <CheckCircle className="w-5 h-5" />
                          <span className="font-medium">Achievement Unlocked!</span>
                        </div>
                      ) : (
                        <div className="text-center text-muted-foreground">
                          <p className="text-sm">
                            {achievement.progress > 0 
                              ? `${achievement.totalRequired - achievement.progress} to go!`
                              : 'Keep learning to unlock this achievement'
                            }
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}